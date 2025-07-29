# %% [markdown]
# # End-to-end physics informed network for quantitative MRI (PINQI)
# A recent DL approach, PINQI, approaches learned quantitative MRI by half quadratic splitting to alternate between two
# subproblems. The first is a linear image reconstruction task
# $$
# \underset{\mathbf{x}}{\min} \frac{1}{2} \| \mathbf{A} \mathbf{x} - \mathbf{y} \|_2^2 + \frac{\lambda_\mathbf{x}}{2} \left\| \mathbf{x} - \mathbf{x}_{\text{reg}} \right\|_2^2 + \frac{\lambda_{\mathbf{q}}}{2} \left\| \mathbf{q}(\mathbf{p}) - \mathbf{x} \right\|_2^2
# $$
# with $\mathbf{x}$ being intermediary qualitative images, $\lambda_{\mathbf{x}}$ and $\lambda_{\mathbf{q}}$ being
# regularization strengths and $\mathbf{x}_{\text{reg}}$ denoting an image prior for regularization.
# The second, non-linear, subproblem is finding the quantitative parameters by solving
# $$
# \underset{\mathbf{p}}{\min} \frac{\lambda_{\mathbf{q}}}{2}\left \| \mathbf{q}(\vec{p}) - \mathbf{x} \right\|_2^2 + \frac{\lambda_{\mathbf{p}}}{2} \left\| \mathbf{p} - \mathbf{p}_{\text{reg}} \right\|_2^2.
# $$
# Here, $\mathbf{p}_{\text{reg}}$ is a prior on the parameter maps and $\lambda_{\mathbf{p}}$ the associated weight for regularization.
# In PINQI, a solution is found by iterating between both subproblems. In each iteration $k=1,\ldots,T$, the image and parameter priors are updated by
# U-Nets. The network parameters and the regularization strengths are trained end-to-end.
# Here, we apply a trained PINQI model to a validation set. We first define the dataset, then define the PINQI model, before loading the model weights
# and applying it to the dataset.

# %% [markdown]
# ## Dataset
# We base the dataset on the BrainWeb phantom (`mrpro.phantoms.brainweb.BrainwebSlices`) and simulate Cartesian random undersampling in phase
# encode direction.

# %%
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypedDict

import einops
import mrpro
import torch


# %%
class BatchType(TypedDict):
    """Typehint for a batch of data."""

    kdata: mrpro.data.KData
    csm: mrpro.data.CsmData
    m0: torch.Tensor
    t1: torch.Tensor
    mask: torch.Tensor


class Dataset(torch.utils.data.Dataset[BatchType]):
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
        self.signalmodel = deepcopy(signalmodel)
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


# %% [markdown]
# ## PINQI
# Next, We define the PINQI model. Here we can make use of the diffferntiable optimization operators in MRpro.


# %%
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
        self.parameter_net = mrpro.nn.nets.UNet(
            dim=2,
            channels_in=n_images * 2,
            channels_out=real_parameters,
            attention_depths=(-1, -2),
            n_features=n_features_parameter_net,
            cond_dim=128,
        )

        self.image_net = mrpro.nn.nets.UNet(
            2, channels_in=2, channels_out=2, attention_depths=(), n_features=n_features_image_net, cond_dim=128
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
        # This can be done once, as the signal model is the same for all samples.

    def get_linear_solver(self, gram: mrpro.operators.LinearOperator):
        """Set up the linear solver."""
        # This needs to be done for each sample, as the undersampling pattern and csm are different for each sample,
        # thus the gram operator of the acquisition operator is different for each sample.

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
        """Get the parameter regularization."""
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
        """Get the image regularization."""
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
        """Estimate the quantitative parameters.

        Parameters
        ----------
        kdata
            The k-space data.
        csm
            The coil sensitivity maps.

        Returns
        -------
        images
            The qualitative images.
        parameters
            The quantitative parameters.
        """
        csm_op = csm.as_operator()
        fourier_op = mrpro.operators.FourierOp.from_kdata(kdata)
        acquisition_op = fourier_op @ csm_op
        gram = acquisition_op.gram
        (zero_filled_image,) = acquisition_op.H(kdata.data)
        images = mrpro.algorithms.optimizers.cg(gram, zero_filled_image, max_iterations=2)
        parameters = self.get_parameter_reg(images, 0)
        linear_solver = self.get_linear_solver(gram)

        for i, (lambda_image, lambda_q, lambda_parameter) in enumerate(self.softplus(self.lambdas_raw)):
            # linear subproblem 1
            image_reg = self.get_image_reg(images, i)
            (signal,) = self.signalmodel(*parameters)
            images = linear_solver(lambda_image, lambda_q, image_reg, signal, zero_filled_image)
            # nonlinear subproblem 2
            parameters_reg = self.get_parameter_reg(images, i + 1)
            parameters = self.nonlinear_solver(lambda_parameter, images, *parameters_reg)
        if self.constraints_op is not None:
            # map the parameters into the constrained space
            parameters = self.constraints_op(*parameters)
        return parameters


# %%
# As a baseline methods for comparision, we use a simple non-learned approach. We reconstruct the qualitative images at different saturation times using iterative SENSE.
# We then perform a  constrained non-linear least squares regression usingL-BFGS to obtain the parameter maps.
# %%
def baseline_solution(
    signalmodel: mrpro.operators.SignalModel,
    constraints_op: mrpro.operators.ConstraintsOp | mrpro.operators.MultiIdentityOp,
    parameter_is_complex: Sequence[bool],
    kdata: mrpro.data.KData,
    csm: mrpro.data.CsmData,
) -> tuple[torch.Tensor, ...]:
    """Compute a baseline solution using SENSE + Regression."""
    sense = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(kdata, csm=csm)
    images = sense(kdata)
    objective = mrpro.operators.functionals.L2NormSquared(images.data) @ signalmodel @ constraints_op
    initial_values = tuple(
        torch.zeros(images.shape[1:], device=images.device, dtype=torch.complex64 if is_complex else torch.float32)
        for is_complex in parameter_is_complex
    )
    solution = constraints_op(*mrpro.algorithms.optimizers.lbfgs(objective, initial_values))
    return solution


# %%
data_folder = Path('/home/zimmer08/.cache/mrpro/brainweb')

signalmodel = mrpro.operators.models.SaturationRecovery((0.5, 1.0, 1.5, 2.0, 8.0))
constraints_op = mrpro.operators.ConstraintsOp(
    bounds=(
        (-2, 2),  # M0 in [-2, 2]
        (0.01, 6.0),  # T1 is constrained between 10 ms and 6 s
    )
)
n_images = len(signalmodel.saturation_time)
parameter_is_complex = [True, False]


dataset = torch.utils.data.Subset(
    Dataset(
        folder=data_folder,
        signalmodel=signalmodel,
        n_images=n_images,
        size=192,
        acceleration=8,
        n_coils=8,
        max_noise=0.05,
        orientation=('axial',),
        random=False,
    ),
    list(range(500)),
)
# %%
checkpoint = torch.load('./examples/scripts/last.ckpt', map_location='cpu')
hyper_parameters = checkpoint['hyper_parameters']


pinqi = PINQI(
    signalmodel=signalmodel,
    constraints_op=constraints_op,
    parameter_is_complex=parameter_is_complex,
    n_images=n_images,
    n_iterations=hyper_parameters['n_iterations'],
    n_features_parameter_net=hyper_parameters['n_features_parameter_net'],
    n_features_image_net=hyper_parameters['n_features_image_net'],
)
state_dict = {
    k.replace('pinqi.', '').replace('_orig_mod.', ''): v
    for k, v in checkpoint['state_dict'].items()
    if 'baseline' not in k
}
pinqi.load_state_dict(state_dict)
# %%
batch = dataset[40]
csm, kdata = batch['csm'], batch['kdata']

if torch.cuda.is_available():
    pinqi, csm, kdata = pinqi.cuda(), csm.cuda(), kdata.cuda()
images, parameters = pinqi(kdata[None], csm[None])
with torch.no_grad():
    predicted_m0, predicted_t1 = (p.cpu().detach().squeeze() for p in parameters[-1])
baseline_m0, baseline_t1 = baseline_solution(signalmodel, constraints_op, parameter_is_complex, kdata, csm)
# %%
(ssim_t1,) = mrpro.operators.functionals.SSIM(batch['t1'][None], batch['mask'][None])(predicted_t1[None])
(mse_t1,) = mrpro.operators.functionals.MSE(batch['t1'], batch['mask'])(predicted_t1)

(mse_baseline,) = mrpro.operators.functionals.MSE(batch['t1'], batch['mask'])(baseline_t1)
nrmse_t1 = torch.sqrt(mse_t1) / batch['t1'][batch['mask']].max()
(ssim_baseline,) = mrpro.operators.functionals.SSIM(batch['t1'][None], batch['mask'][None])(baseline_t1[None])
nrmse_baseline = torch.sqrt(mse_baseline) / batch['t1'][batch['mask']].max()


# %%
import matplotlib.pyplot as plt
from cmap import Colormap

cmap = Colormap('lipari').to_matplotlib()

print(f'SSIM: {ssim_baseline.item():.4f}, NRMSE: {nrmse_baseline.item():.4f}')
print(f'SSIM: {ssim_t1.item():.4f}, NRMSE: {nrmse_t1.item():.4f}')


fig, ax = plt.subplots(
    1,
    5,
    gridspec_kw={
        'width_ratios': [1, 1, 1, 0.28, 0.075],
        'wspace': -0.25,
    },
    figsize=(6.5, 2.5),
)
baseline_t1 = baseline_t1.squeeze()
baseline_t1[~batch['mask']] = torch.nan
ax[0].imshow(baseline_t1, vmin=0, vmax=2, cmap=cmap)
ax[0].axis('off')
ax[0].set_title('SENSE + NLS')
ax[0].text(
    0.5,
    -0.00,
    f'SSIM: {ssim_baseline.item():.2f}',
    color='black',
    horizontalalignment='center',
    verticalalignment='top',
    transform=ax[0].transAxes,
    size=11,
)
predicted_t1 = predicted_t1.squeeze()
predicted_t1[~batch['mask']] = torch.nan
ax[1].imshow(predicted_t1, vmin=0, vmax=2, cmap=cmap)
ax[1].axis('off')
ax[1].set_title('PINQI')
ax[1].text(
    0.5,
    -0.0,
    f'SSIM: {ssim_t1.item():.2f}',
    color='black',
    horizontalalignment='center',
    verticalalignment='top',
    transform=ax[1].transAxes,
    size=11,
)

target_t1 = batch['t1'].squeeze()
target_t1[~batch['mask']] = torch.nan
im = ax[2].imshow(target_t1, vmin=0, vmax=2, cmap=cmap)
ax[2].axis('off')
ax[2].set_title(
    'Ground Truth',
)
ax[-2].axis('off')
fig.tight_layout()
plt.colorbar(im, cax=ax[-1], label='$T_1$ (s)')
fig.savefig('/home/zimmer08/code/mrpro/examples/scripts/pinqi_t1_3.pdf', bbox_inches='tight', pad_inches=0)


# %%

1
# %%
# %%
