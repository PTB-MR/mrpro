# ruff: noqa: D102, ANN201

import collections
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

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
    """Typehint for a batch of data."""

    kdata: mrpro.data.KData
    csm: mrpro.data.CsmData
    m0: torch.Tensor
    t1: torch.Tensor
    mask: torch.Tensor


class Dataset(torch.utils.data.Dataset):
    """A brainweb based cartesian qMRI dataset."""

    def __init__(
        self,
        folder: Path,
        signalmodel: mrpro.operators.SignalModel,
        n_images: int,
        size: int,
        acceleration: int,
        n_coils: int,
        max_noise: float,
        orientation: Sequence[Literal['axial', 'coronal', 'sagittal']],
        random: bool = True,
    ):
        """Initialize the dataset."""
        if random:
            augment = mrpro.phantoms.brainweb.augment(size=size)
        else:
            augment = mrpro.phantoms.brainweb.augment(
                size=size,
                max_random_shear=0,
                max_random_rotation=0,
                max_random_scaling_factor=0,
                p_horizontal_flip=0,
                p_vertical_flip=1.0,
            )
        self.phantom = mrpro.phantoms.brainweb.BrainwebSlices(
            folder=folder,
            what=('m0', 't1', 'mask'),
            seed='index' if not random else 'random',
            slice_preparation=augment,
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
        """Get the length of the dataset."""
        return len(self.phantom)

    def __getitem__(self, index: int):
        """Get an item from the dataset."""
        phantom = self.phantom[index]
        (images,) = self.signalmodel(phantom['m0'], phantom['t1'])
        seed = int(torch.randint(0, 1000000, (1,))) if self._random else index

        traj = mrpro.data.traj_calculators.KTrajectoryCartesian.gaussian_variable_density(
            encoding_matrix=self.encoding_matrix,
            seed=seed,
            acceleration=self.acceleration,
            fwhm_ratio=1.5,
            n_center=10,
            n_other=(self._n_images,),
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

        fourier_op = mrpro.operators.FourierOp(self.encoding_matrix, self.encoding_matrix, traj)
        csm = mrpro.data.CsmData(
            mrpro.phantoms.coils.birdcage_2d(self.n_coils, self.encoding_matrix),
            header,
        )
        images = einops.rearrange(images, 't y x -> t 1 1 y x')
        (data,) = (fourier_op @ csm.as_operator())(images)
        data = data + torch.randn_like(data) * torch.rand(1) * self.max_noise * data.std()
        kdata = mrpro.data.KData(header, data, traj)
        return {'kdata': kdata, 'csm': csm, **phantom}


def collate_fn(batch: Any):  # noqa: ANN401
    """Join dataclasses to a batch."""
    return torch.utils.data._utils.collate.collate(
        batch,
        collate_fn_map={
            mrpro.data.Dataclass: lambda batch, *, collate_fn_map: batch[0].stack(*batch[1:]),  # noqa: ARG005
            **torch.utils.data._utils.collate.default_collate_fn_map,
        },
    )


class PINQI(torch.nn.Module):
    """PINQI model."""

    def __init__(
        self,
        signalmodel: mrpro.operators.SignalModel,
        constraints_op: mrpro.operators.ConstraintsOp | mrpro.operators.MultiIdentityOp,
        parameter_is_complex: Sequence[bool],
        n_images: int,
        n_iterations: int,
        n_features_parameter_net: Sequence[int],
        n_features_image_net: Sequence[int],
    ):
        """Initialize the PINQI model."""
        super().__init__()
        self.signalmodel = mrpro.operators.RearrangeOp('t batch ... -> batch t ...') @ signalmodel @ constraints_op
        self.constraints_op = constraints_op
        self._n_images = n_images
        self._parameter_is_complex = parameter_is_complex
        real_parameters = sum(1 for c in parameter_is_complex if c) + len(parameter_is_complex)
        self.parameter_net = torch.compile(
            mrpro.nn.nets.UNet(
                dim=2,
                channels_in=n_images * 2,
                channels_out=real_parameters,
                attention_depths=(-1, -2),
                n_features=n_features_parameter_net,
                cond_dim=128,
            ),
            dynamic=False,
            fullgraph=True,
        )
        self.image_net = torch.compile(
            mrpro.nn.nets.UNet(
                2, channels_in=2, channels_out=2, attention_depths=(), n_features=n_features_image_net, cond_dim=128
            ),
            dynamic=False,
            fullgraph=True,
        )
        self.lambdas_raw = torch.nn.Parameter(torch.ones(n_iterations, 3))
        self.softplus = torch.nn.Softplus(beta=5)
        self.iteration_embedding = torch.nn.Embedding(n_iterations + 1, 128)

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

    def get_parameter_reg(self, image: torch.Tensor, iteration: int = 0) -> tuple[torch.Tensor, ...]:
        image = einops.rearrange(
            torch.view_as_real(image),
            'batch t 1 1 y x complex-> batch (t complex) y x',
        )
        cond = self.iteration_embedding(torch.tensor(iteration, device=image.device))[None]
        parameters = self.parameter_net(image.contiguous(), cond=cond)
        parameters = einops.rearrange(parameters, 'batch parameters y x-> parameters batch 1 1 y x')
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

    def get_image_reg(self, image: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        batch = image.shape[0]
        image = einops.rearrange(
            torch.view_as_real(image),
            'batch t 1 1 y x complex-> (batch t) complex y x',
        )
        cond = self.iteration_embedding(torch.tensor(iteration, device=image.device))[None]
        image = image + self.image_net(image.contiguous(), cond=cond)
        image = einops.rearrange(image, '(batch t) complex y x-> batch t 1 1 y x complex', batch=batch)
        return torch.view_as_complex(image.contiguous())

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData):
        csm_op = csm.as_operator()
        fourier_op = mrpro.operators.FourierOp.from_kdata(kdata)
        acquisition_op = fourier_op @ csm_op
        gram = acquisition_op.gram
        (zero_filled_image,) = acquisition_op.H(kdata.data)
        images = list(mrpro.algorithms.optimizers.cg(gram, zero_filled_image, max_iterations=2))
        parameters = [self.get_parameter_reg(images[-1], 0)]
        linear_solver = self.get_linear_solver(gram)

        for i, (lambda_image, lambda_q, lambda_parameter) in enumerate(self.softplus(self.lambdas_raw)):
            image_reg = self.get_image_reg(images[-1], i + 1)
            (signal,) = self.signalmodel(*parameters[-1])
            images.extend(linear_solver(lambda_image, lambda_q, image_reg, signal, zero_filled_image))
            parameters_reg = self.get_parameter_reg(images[-1], i + 1)
            parameters.append(self.nonlinear_solver(lambda_parameter, images[-1], *parameters_reg))
        if self.constraints_op is not None:
            parameters = [self.constraints_op(*p) for p in parameters]
        return images, parameters


class DataModule(pl.LightningDataModule):
    """Data module for training the PINQI model."""

    def __init__(
        self,
        folder: Path,
        signalmodel: mrpro.operators.SignalModel,
        n_images: int,
        size: int = 192,
        acceleration: int = 10,
        n_coils: int = 8,
        max_noise: float = 0.1,
        orientation_train: Sequence[Literal['axial', 'coronal', 'sagittal']] = (
            'axial',
            'coronal',
            'sagittal',
        ),
        orientation_val: Sequence[Literal['axial', 'coronal', 'sagittal']] = ('axial',),
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """Initialize the data module."""
        super().__init__()
        self.save_hyperparameters(ignore=['signalmodel', 'folder', 'num_workers'])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = Dataset(
            folder=folder,
            signalmodel=signalmodel,
            n_images=n_images,
            size=size,
            acceleration=acceleration,
            n_coils=n_coils,
            max_noise=max_noise,
            orientation=orientation_train,
            random=True,
        )
        self.val_dataset = torch.utils.data.Subset(
            Dataset(
                folder=folder,
                signalmodel=signalmodel,
                n_images=n_images,
                size=size,
                acceleration=acceleration,
                n_coils=n_coils,
                max_noise=max_noise,
                orientation=orientation_val,
                random=False,
            ),
            list(range(30, 500, 20)),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            worker_init_fn=lambda *_: torch.set_num_threads(1),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )


class PinqiModule(pl.LightningModule):
    """Module for training the PINQI model."""

    def __init__(
        self,
        signalmodel: mrpro.operators.SignalModel,
        constraints_op: mrpro.operators.ConstraintsOp,
        parameter_is_complex: Sequence[bool],
        n_images: int,
        n_iterations: int = 4,
        n_features_parameter_net: Sequence[int] = (64, 128, 192, 256),
        n_features_image_net: Sequence[int] = (32, 48, 64, 96),
        lr: float = 3e-4,  # noqa: ARG002
        weight_decay: float = 1e-3,  # noqa: ARG002
        loss_weights: Sequence[float] = (0.2, 0.1, 0.1, 0.1, 0.8),
    ):
        """Initialize the PINQI module."""
        super().__init__()
        self.save_hyperparameters(ignore=['signalmodel', 'constraints_op'])
        if len(loss_weights) != n_iterations + 1:
            raise ValueError(f'loss_weights must be of length {n_iterations + 1} for {n_iterations} iterations')
        signalmodel, constraints_op = map(deepcopy, (signalmodel, constraints_op))
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
        self.baseline = Baseline(signalmodel, constraints_op, parameter_is_complex)

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData):
        """Apply the PINQI model to the data."""
        return self.pinqi(kdata, csm)

    def loss(self, predictions: Sequence[torch.Tensor], batch: BatchType) -> torch.Tensor:
        """Compute the loss."""
        loss = torch.tensor(0.0, device=self.device)
        target_m0, target_t1, mask = map(torch.squeeze, (batch['m0'], batch['t1'], batch['mask']))
        for prediction, weight in zip(predictions, self.hparams.loss_weights, strict=False):
            prediction_m0, prediction_t1 = map(torch.squeeze, prediction)
            loss_t1 = torch.nn.functional.mse_loss(prediction_t1[mask], target_t1[mask])
            loss_m0 = torch.nn.functional.mse_loss(
                torch.view_as_real(prediction_m0[mask]),
                torch.view_as_real(target_m0[mask]),
            )
            loss_outside = prediction_m0[~mask].abs().mean()
            loss = loss + weight * (loss_t1 + 0.5 * loss_m0 + 0.1 * loss_outside)
        return loss

    def training_step(self, batch: BatchType, _batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, parameters = self(batch['kdata'], batch['csm'])
        loss = self.loss(parameters, batch)
        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch['mask']),
        )
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> None:
        """Validate.

        Needs to be adapted for other signal models than Saturation Recovery.
        """
        images, parameters = self(batch['kdata'], batch['csm'])
        loss = self.loss(parameters, batch)

        pred_m0, pred_t1 = parameters[-1]
        target_t1, target_m0 = batch['t1'], batch['m0']
        mask = batch['mask']
        batch_size = len(batch['mask'])
        (ssim_t1,) = mrpro.operators.functionals.SSIM(target_t1, mask)(pred_t1)
        (l1_t1,) = mrpro.operators.functionals.L1Norm(target_t1, mask)(pred_t1)
        (l1_m0,) = mrpro.operators.functionals.L1Norm(target_m0, mask)(pred_m0)
        self.log('val/ssim_t1', ssim_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/l1_t1', l1_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/l1_m0', l1_m0, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True, batch_size=batch_size)

        if batch_idx == 0:
            self.validation_step_outputs['target_t1'].append(batch['t1'])
            self.validation_step_outputs['pred_t1'].append(pred_t1)
            self.validation_step_outputs['pred_m0'].append(pred_m0)
            self.validation_step_outputs['target_m0'].append(target_m0)
            self.validation_step_outputs['mask'].append(batch['mask'])
            baseline_m0, baseline_t1 = self.baseline(batch['kdata'], batch['csm'])
            self.validation_step_outputs['baseline_t1'].append(baseline_t1)
            self.validation_step_outputs['baseline_m0'].append(baseline_m0)

    def on_validation_epoch_end(self):
        """Validate.

        Needs to be adapted for other signal models than Saturation Recovery.
        """
        outputs = {k: torch.cat(v) for k, v in self.validation_step_outputs.items()}
        self.validation_step_outputs.clear()
        outputs = cast(dict[str, torch.Tensor], self.all_gather(outputs))

        if not self.trainer.is_global_zero:
            return
        outputs = {k: v.flatten(0, 1).cpu() for k, v in outputs.items()}

        samples = len(outputs['mask'])
        fig, axes = plt.subplots(4, samples, figsize=(4 * samples, 16))

        for i in range(samples):
            self.result_plot(
                outputs['target_t1'][i],
                outputs['pred_t1'][i],
                outputs['mask'][i],
                axes[:, i],
                outputs['baseline_t1'][i],
                '$T_1$ (s)',
            )
        fig.suptitle(f'$T_1$ Epoch {self.current_epoch}')
        self.logger.run['val/images/t1'].log(fig)
        plt.close(fig)

        fig, axes = plt.subplots(4, samples, figsize=(4 * samples, 12))
        for i in range(samples):
            self.result_plot(
                outputs['target_m0'][i].abs(),
                outputs['pred_m0'][i].abs(),
                outputs['mask'][i],
                axes[:, i],
                outputs['baseline_m0'][i].abs(),
                '$|M_0|$ (a.u.)',
            )
        fig.suptitle(f'$|M_0|$ Epoch {self.current_epoch}')
        self.logger.run['val/images/m0'].log(fig)
        plt.close(fig)

    def result_plot(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        axes: Sequence[plt.Axes],
        baseline: torch.Tensor,
        label: str,
    ) -> None:
        """Plot the results."""
        target = target.squeeze().numpy()
        pred = pred.squeeze().detach().numpy()
        mask = mask.squeeze().detach().numpy().astype(bool)
        baseline = baseline.squeeze().detach().numpy()

        target[~mask] = np.nan
        pred[~mask] = np.nan
        baseline[~mask] = np.nan
        difference = (target - pred) / target * 100
        vmax = np.nanmax(target)

        im0 = axes[0].imshow(target, vmin=0, vmax=vmax)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=label)

        im1 = axes[1].imshow(baseline, vmin=0, vmax=vmax)
        axes[1].set_title('SENSE + Regression')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=label)

        im2 = axes[2].imshow(pred, vmin=0, vmax=vmax)
        axes[2].set_title('PINQI')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label=label)

        diff_vmax = np.nanpercentile(np.abs(difference), 90)
        im3 = axes[3].imshow(difference, cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax)
        axes[3].set_title('rel. Error')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label='%')

    def configure_optimizers(
        self,
    ) -> dict:
        """Configure the optimizer and the learning rate scheduler."""
        scalars = ('lambdas_raw', 'rezero')
        params, scalar_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(s in n for s in scalars):
                scalar_params.append(p)
            else:
                params.append(p)
        optimizer = torch.optim.AdamW(
            [
                {'params': params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr},
                {'params': scalar_params, 'weight_decay': 0.0, 'lr': self.hparams.lr * 10},
            ],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.hparams.lr, 10 * self.hparams.lr],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            div_factor=20,
            final_div_factor=300,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


class Baseline(torch.nn.Module):
    """Baseline solution using SENSE + Regression."""

    def __init__(
        self,
        signalmodel: mrpro.operators.SignalModel,
        constraints_op: mrpro.operators.ConstraintsOp | mrpro.operators.MultiIdentityOp,
        parameter_is_complex: Sequence[bool],
    ):
        """Initialize the baseline."""
        super().__init__()
        self.signalmodel = signalmodel
        self.constraints_op = constraints_op
        self.parameter_is_complex = parameter_is_complex

    def forward(self, kdata: mrpro.data.KData, csm: mrpro.data.CsmData) -> tuple[torch.Tensor, ...]:
        """Compute the baseline solution."""
        sense = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(kdata, csm=csm)
        images = sense(kdata).rearrange('batch time ...-> time batch ...')

        objective = mrpro.operators.functionals.L2NormSquared(images.data) @ self.signalmodel @ self.constraints_op
        initial_values = tuple(
            torch.zeros(images.shape[1:], device=images.device, dtype=torch.complex64 if is_complex else torch.float32)
            for is_complex in self.parameter_is_complex
        )
        solution = self.constraints_op(*mrpro.algorithms.optimizers.lbfgs(objective, initial_values))
        return solution


class LogLambdasCallback(pl.Callback):
    """Log the lambdas."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: PinqiModule,
        _outputs: dict,
        _batch: BatchType,
        _batch_idx: int,
    ) -> None:
        if trainer.global_step % 10 == 0:
            lambdas = pl_module.pinqi.softplus(pl_module.pinqi.lambdas_raw).detach().cpu().numpy()
            for iteration, (lambda_image, lambda_q, lambda_parameter) in enumerate(lambdas):
                self.log_dict(
                    {
                        f'parameter/lambda_image_{iteration}': lambda_image,
                        f'parameter/lambda_q_{iteration}': lambda_q,
                        f'parameter/lambda_parameter_{iteration}': lambda_parameter,
                    },
                    on_step=True,
                    on_epoch=False,
                )


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.compile_threads = 4
    torch._inductor.config.worker_start_method = 'fork'
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.cache_size_limit = 256
    torch._functorch.config.activation_memory_budget = 0.95

    data_folder = Path('/scratch/zimmer08/brainweb')

    signalmodel = mrpro.operators.models.SaturationRecovery((0.5, 1.0, 1.5, 2.0, 8.0))
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
        num_workers=16,
        size=192,
        acceleration=8,
        n_coils=8,
        max_noise=0.1,
    )

    model = PinqiModule(
        signalmodel=signalmodel,
        constraints_op=constraints_op,
        parameter_is_complex=parameter_is_complex,
        n_images=n_images,
    )

    neptune_logger = NeptuneLogger(
        log_model_checkpoints=False,
        dependencies='infer',
    )
    neptune_logger.log_model_summary(model=model, max_depth=-1)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=2,
        dirpath=Path('checkpoints') / str(neptune_logger.version),
        filename='{epoch:02d}-{val/loss:.4f}',
        save_last=True,
    )

    strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=4,
        strategy=strategy,
        logger=neptune_logger,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
            LogLambdasCallback(),
        ],
        log_every_n_steps=10,
        gradient_clip_algorithm='norm',
        gradient_clip_val=5.0,
    )

    trainer.fit(model, datamodule=dm)
