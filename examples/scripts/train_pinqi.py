# ruff: noqa: D102, ANN201

import collections
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypedDict, cast

import einops
import matplotlib.pyplot as plt
import mrpro
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data._utils
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.strategies import DDPStrategy


# mrpro.phantoms.brainweb.download_brainweb(workers=2, progress=True)


class BatchType(TypedDict):
    kdata: mrpro.data.KData
    csm: mrpro.data.CsmData
    m0: torch.Tensor
    t1: torch.Tensor
    mask: torch.Tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder: Path,
        signalmodel: mrpro.operators.SignalModel,
        n_images: int,
        size: int = 192,
        acceleration: int = 10,
        n_coils: int = 8,
        random: bool = True,
        max_noise: float = 0.1,
        orientation: Sequence[Literal["axial", "coronal", "sagittal"]] = (
            "axial",
            "coronal",
            "sagittal",
        ),
    ):
        self.phantom = mrpro.phantoms.brainweb.BrainwebSlices(
            folder=folder,
            what=("m0", "t1", "mask"),
            seed="index" if not random else "random",
            slice_preparation=mrpro.phantoms.brainweb.augment(size=size),
            orientation=orientation,
        )
        self.signalmodel = signalmodel
        self.encoding_matrix = mrpro.data.SpatialDimension(1, size, size)
        self.fov = mrpro.data.SpatialDimension(0.01, 0.25, 0.25)
        self.acceleration = acceleration
        self.n_coils = n_coils
        self._random = random
        self.max_noise = max_noise
        self._n_images = n_images

    def __len__(self) -> int:
        return len(self.phantom)

    def __getitem__(self, index: int):
        phantom = self.phantom[index]
        (images,) = self.signalmodel(phantom["m0"], phantom["t1"])
        seed = int(torch.randint(0, 1000000, (1,))) if self._random else index

        traj = (
            mrpro.data.traj_calculators.KTrajectoryCartesian.gaussian_variable_density(
                encoding_matrix=self.encoding_matrix,
                seed=seed,
                acceleration=self.acceleration,
                fwhm_ratio=2,
                n_center=8,
                n_other=(self._n_images,),
            )
        )
        header = mrpro.data.KHeader(
            encoding_matrix=self.encoding_matrix,
            recon_matrix=self.encoding_matrix,
            recon_fov=self.fov,
            encoding_fov=self.fov,
        )

        if isinstance(self.signalmodel, mrpro.operators.models.SaturationRecovery):
            header.ti = self.signalmodel.saturation_time.tolist()
        elif isinstance(self.signalmodel, mrpro.operators.models.InversionRecovery):
            header.ti = self.signalmodel.ti.tolist()

        fourier_op = mrpro.operators.FourierOp(
            self.encoding_matrix, self.encoding_matrix, traj
        )
        csm = mrpro.data.CsmData(
            mrpro.phantoms.coils.birdcage_2d(self.n_coils, self.encoding_matrix),
            header,
        )
        images = einops.rearrange(images, "t y x -> t 1 1 y x")
        (data,) = (fourier_op @ csm.as_operator())(images)
        data = (
            data + torch.randn_like(data) * torch.rand(1) * self.max_noise * data.std()
        )
        kdata = mrpro.data.KData(header, data, traj)
        return {"kdata": kdata, "csm": csm, **phantom}

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data._utils.collate.collate(
            batch,
            collate_fn_map={
                mrpro.data.Dataclass: lambda batch, *, _: batch[0].stack(*batch[1:]),
                **torch.utils.data._utils.collate.default_collate_fn_map,
            },
        )


class PINQI(torch.nn.Module):
    def __init__(
        self,
        signalmodel: mrpro.operators.SignalModel,
        constraints_op: mrpro.operators.ConstraintsOp | mrpro.operators.MultiIdentityOp,
        parameter_is_complex: Sequence[bool],
        n_images: int,
        n_iterations: int = 4,
        n_features_parameter_net: Sequence[int] = (64, 128, 192, 256),
        n_features_image_net: Sequence[int] = (16, 32, 48, 64),
    ):
        super().__init__()
        self.signalmodel = (
            mrpro.operators.RearrangeOp("t batch ... -> batch t ...")
            @ deepcopy(signalmodel)
            @ constraints_op
        )
        self.constraints_op = constraints_op
        self._n_images = n_images
        self._parameter_is_complex = parameter_is_complex
        real_parameters = sum(1 for c in parameter_is_complex if c) + len(
            parameter_is_complex
        )
        self.parameter_net = mrpro.nn.nets.UNet(
            dim=2,
            channels_in=n_images * 2,
            channels_out=real_parameters,
            attention_depths=(-1,),
            n_features=n_features_parameter_net,
        )
        self.image_net = mrpro.nn.nets.UNet(
            2,
            channels_in=2,
            channels_out=2,
            attention_depths=(),
            n_features=n_features_image_net,
        )
        self.lambdas_raw = torch.nn.Parameter(torch.ones(n_iterations, 3))
        self.softplus = torch.nn.Softplus()

        def objective_factory(
            lambda_parameters: torch.Tensor,
            image: torch.Tensor,
            *parameter_reg: torch.Tensor,
        ):
            dc = mrpro.operators.functionals.L2NormSquared(image) @ self.signalmodel
            reg = mrpro.operators.ProximableFunctionalSeparableSum(
                *[mrpro.operators.functionals.L2NormSquared(r) for r in parameter_reg]
            )
            return dc + lambda_parameters * reg

        self.nonlinear_solver = mrpro.operators.OptimizerOp(
            objective_factory,
            lambda _l, _i, *parameter_reg: parameter_reg,
        )

    def get_linear_solver(self, gram: mrpro.operators.LinearOperator):
        def operator_factory(
            lambda_image: torch.Tensor,
            lambda_q: torch.Tensor,
            *_,
        ):
            return gram + lambda_image + lambda_q

        def rhs_factory(
            lambda_image: torch.Tensor,
            lambda_q: torch.Tensor,
            image_reg: torch.Tensor,
            signal: torch.Tensor,
            zero_filled_image: torch.Tensor,
        ):
            return (zero_filled_image + lambda_image * image_reg + lambda_q * signal,)

        return mrpro.operators.ConjugateGradientOp(
            operator_factory=operator_factory,
            rhs_factory=rhs_factory,
        )

    def get_parameter_reg(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        image = einops.rearrange(
            torch.view_as_real(image),
            "batch t 1 1 y x complex-> batch (t complex) y x",
        )
        parameters = self.parameter_net(image.contiguous())
        parameters = einops.rearrange(
            parameters, "batch parameters y x-> parameters batch 1 1 y x"
        )
        i = 0
        result = []
        for is_complex in self._parameter_is_complex:
            if is_complex:
                result.append(torch.complex(parameters[i], parameters[i + 1]))
                i += 2
            else:
                result.append(parameters[i])
                i += 1
        return tuple(result)

    def get_image_reg(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        image = einops.rearrange(
            torch.view_as_real(image),
            "batch t 1 1 y x complex-> (batch t) complex y x",
        )
        image = image + self.image_net(image.contiguous())
        image = einops.rearrange(
            image, "(batch t) complex y x-> batch t 1 1 y x complex", batch=batch
        )
        return torch.view_as_complex(image.contiguous())

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData):
        csm_op = csm.as_operator()
        fourier_op = mrpro.operators.FourierOp.from_kdata(kdata)
        acquisition_op = fourier_op @ csm_op
        gram = acquisition_op.gram
        (zero_filled_image,) = acquisition_op.H(kdata.data)
        images = list(
            mrpro.algorithms.optimizers.cg(gram, zero_filled_image, max_iterations=2)
        )
        parameters = [self.get_parameter_reg(images[-1])]
        linear_solver = self.get_linear_solver(gram)

        for lambda_image, lambda_q, lambda_parameter in self.softplus(self.lambdas_raw):
            image_reg = self.get_image_reg(images[-1])
            (signal,) = self.signalmodel(*parameters[-1])
            images.extend(
                linear_solver(
                    lambda_image, lambda_q, image_reg, signal, zero_filled_image
                )
            )
            parameters_reg = self.get_parameter_reg(images[-1])
            parameters.append(
                self.nonlinear_solver(lambda_parameter, images[-1], *parameters_reg)
            )
        if self.constraints_op is not None:
            parameters = [self.constraints_op(*p) for p in parameters]
        return images, parameters


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        folder: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        signalmodel: mrpro.operators.SignalModel = mrpro.operators.models.SaturationRecovery(
            (0.5, 1.0, 1.5, 2.0, 6.0)
        ),
        n_images: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = Dataset(
            folder=folder,
            signalmodel=signalmodel,
            n_images=n_images,
            **kwargs,
            random=True,
        )
        self.val_dataset = Dataset(
            folder=folder,
            orientation=("axial",),
            signalmodel=signalmodel,
            n_images=n_images,
            **kwargs,
            random=False,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train_dataset.collate_fn,
            worker_init_fn=lambda *_: torch.set_num_threads(1),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.val_dataset.collate_fn,
        )


class Module(pl.LightningModule):
    def __init__(
        self,
        signalmodel: mrpro.operators.SignalModel,
        constraints_op: mrpro.operators.ConstraintsOp,
        parameter_is_complex: Sequence[bool],
        n_images: int,
        n_iterations: int = 4,
        n_features_parameter_net: Sequence[int] = (64, 128, 192, 256),
        n_features_image_net: Sequence[int] = (16, 32, 48, 64),
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        loss_weights: Sequence[float] = (0.1, 0.1, 0.1, 0.2, 0.5),
    ):
        super().__init__()
        self.save_hyperparameters()
        if len(loss_weights) != n_iterations + 1:
            raise ValueError(
                f"loss_weights must be of length {n_iterations + 1} for {n_iterations} iterations"
            )

        self.pinqi = PINQI(
            signalmodel=signalmodel,
            constraints_op=constraints_op,
            parameter_is_complex=parameter_is_complex,
            n_images=n_images,
            n_iterations=n_iterations,
            n_features_parameter_net=n_features_parameter_net,
            n_features_image_net=n_features_image_net,
        )

        self.validation_step_outputs = collections.defaultdict(list)

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData):
        return self.pinqi(kdata, csm)

    def loss(self, predictions, batch):
        loss = torch.tensor(0.0, device=self.device)
        target_m0 = batch["m0"]
        target_t1 = batch["t1"]
        mask = batch["mask"]
        for prediction, weight in zip(
            predictions, self.hparams.loss_weights, strict=False
        ):
            prediction_m0, prediction_t1 = prediction
            loss_t1 = torch.nn.functional.mse_loss(
                prediction_t1.squeeze()[mask], target_t1[mask]
            )
            loss_m0 = torch.nn.functional.mse_loss(
                torch.view_as_real((prediction_m0).squeeze()[mask]),
                torch.view_as_real(target_m0[mask]),
            )
            loss_outside = prediction_m0[~mask].abs().mean()
            loss = loss + weight * (loss_t1 + 0.5 * loss_m0 + 0.1 * loss_outside)
        return loss

    def training_step(self, batch, batch_idx):
        images, parameters = self(batch["kdata"], batch["csm"])
        loss = self.loss(parameters, batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> None:
        images, parameters = self(batch["kdata"], batch["csm"])
        loss = self.loss(parameters, batch)

        pred_m0, pred_t1 = parameters[-1]
        target_t1, target_m0 = batch["t1"], batch["m0"]
        mask = batch["mask"]
        (ssim_t1,) = mrpro.operators.functionals.SSIM(target_t1, mask)(pred_t1)
        (l1_t1,) = mrpro.operators.functionals.L1Norm(target_t1, mask)(pred_t1)
        (l1_m0,) = mrpro.operators.functionals.L1Norm(target_m0, mask)(pred_m0)
        self.log("val/ssim_t1", ssim_t1, on_epoch=True, sync_dist=True)
        self.log("val/l1_t1", l1_t1, on_epoch=True, sync_dist=True)
        self.log("val/l1_m0", l1_m0, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self.validation_step_outputs["target_t1"].append(batch["t1"])
            self.validation_step_outputs["pred_t1"].append(pred_t1)
            self.validation_step_outputs["pred_m0"].append(pred_m0)
            self.validation_step_outputs["target_m0"].append(target_m0)
            self.validation_step_outputs["mask"].append(batch["mask"])

    def on_validation_epoch_end(self):
        outputs = {k: torch.cat(v) for k, v in self.validation_step_outputs.items()}
        self.validation_step_outputs.clear()
        outputs = cast(dict[str, torch.Tensor], self.all_gather(outputs))

        if not self.trainer.is_global_zero:
            return
        outputs = {k: v.flatten(0, 1).cpu() for k, v in outputs.items()}

        samples = 4
        fig, axes = plt.subplots(3, samples, figsize=(4 * samples, 12))
        for i in range(samples):
            self.result_plot(
                outputs["target_t1"][i],
                outputs["pred_t1"][i],
                outputs["mask"][i],
                axes[:, i],
            )
        fig.suptitle(f"T1 Epoch {self.current_epoch}")
        self.logger.run["val/images/t1"].log(fig)
        plt.close(fig)

        fig, axes = plt.subplots(3, samples, figsize=(4 * samples, 12))
        for i in range(samples):
            self.result_plot(
                outputs["target_m0"][i].abs(),
                outputs["pred_m0"][i].abs(),
                outputs["mask"][i],
                axes[:, i],
            )
        fig.suptitle(f"|M0| Epoch {self.current_epoch}")
        self.logger.run["val/images/m0"].log(fig)
        plt.close(fig)

    def result_plot(self, target, pred, mask, axes):
        target = target.squeeze().numpy()
        pred = pred.squeeze().detach().numpy()
        mask = mask.squeeze().detach().numpy().astype(bool)

        target[~mask] = np.nan
        pred[~mask] = np.nan
        difference = target - pred
        vmax = np.nanmax(target)

        im1 = axes[0].imshow(target, vmin=0, vmax=vmax)
        axes[0].set_title("Target")
        axes[0].axis("off")
        axes[0].colorbar(im1)

        im2 = axes[1].imshow(pred, vmin=0, vmax=vmax)
        axes[1].set_title("Predicted")
        axes[1].axis("off")
        axes[1].colorbar(im2)

        diff_vmax = np.nanmax(np.abs(difference))
        im3 = axes[2].imshow(
            difference, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax
        )
        axes[2].set_title("Difference")
        axes[2].axis("off")
        axes[2].colorbar(im3)
        return axes

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.max_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=200,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


# %%
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch._inductor.config.worker_start_method = "fork"
    torch._inductor.config.compile_threads = 4
    torch._dynamo.config.capture_scalar_outputs = True
    torch._functorch.config.activation_memory_budget = 0.9
    torch._dynamo.config.cache_size_limit = 256

    data_folder = Path("/scratch/zimmer08/brainweb")

    signalmodel = mrpro.operators.models.SaturationRecovery((0.5, 1.0, 1.5, 2.0, 6.0))
    constraints_op = mrpro.operators.ConstraintsOp(
        bounds=(
            (-2, 2),  # M0 in [-2, 2]
            (0.01, 6.0),  # T1 is constrained between 10 ms and 6 s
        )
    )
    n_images = len(signalmodel.saturation_time)
    parameter_is_complex = [True, False]

    dm = DataModule(
        folder=data_folder,
        signalmodel=signalmodel,
        n_images=n_images,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        size=192,
        acceleration=10,
        n_coils=8,
        max_noise=0.1,
    )

    model = Module(
        signalmodel=signalmodel,
        constraints_op=constraints_op,
        parameter_is_complex=parameter_is_complex,
        n_images=n_images,
        lr=3e-4,
        weight_decay=1e-4,
        n_iterations=4,
    )

    neptune_logger = NeptuneLogger(
        log_model_checkpoints=False,
        dependencies="infer",
    )
    neptune_logger.log_hyperparams(model.hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        dirpath=Path("checkpoints") / str(neptune_logger.version),
        filename="{epoch:02d}-{val/loss:.4f}",
        save_last=True,
    )

    strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=4,
        strategy=strategy,
        logger=neptune_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
        ],
        log_every_n_steps=10,
        precision="16-mixed",
    )

    trainer.fit(model, datamodule=dm)
