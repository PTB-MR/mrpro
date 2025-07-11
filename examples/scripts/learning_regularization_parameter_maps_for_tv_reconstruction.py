# %% [markdown]
# # Learning spatially adaptive regularization parameter maps for total-variation (TV)-minimization reconstruction

# %% [markdown]
# ## Overview
# In this notebook, we demonstrate how the method presented in
# [[Kofler et al, SIIMS 2023](https://epubs.siam.org/doi/abs/10.1137/23M1552486)] can be implemented for
# 2D MR reconstruction problems using MRpro. The method consists of two main blocks.
# 1) The first block is a neural network architecture that estimates spatially adaptive regularization
# parameter maps from an input image, which are then used in 2).
# 2) The second block corresponds to an unrolled unrolled Primal-Dual Hybrid Gradient (PDHG) algorithm
# [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)] that
# reconstructs the image from the undersampled k-space data measurements assuming the regularization
# parameter maps to be fixed.
#
# The entire network can be trained end-to-end using MRpro, allowing to learn to
# estimate spatially adaptive regularization parameter maps for TV from an input image.

# %% [markdown]
# ## The method
# In the TV-example, (see <project:tv_minimization_reconstruction_pdhg.ipynb>), you can see how to employ the primal
# dual hybrid gradient (PDHG - `mrpro.algorithms.optimizers.pdhg`) method
# [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)]
# to solve the TV-minimization problem. For data acquired according to the usual model
#
# $ y = Ax_{\mathrm{true}} + n, $
#
# where $A$ contains the Fourier transform and the coil sensitivity maps operator, etc, and $n$ is complex-valued
# Gaussian noise, the TV-minimization problem is given by
#
# $\mathcal{F}_{\lambda}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| \nabla x \|_1, \quad \quad \quad (1)$
#
# where $\nabla$ is the discretized gradient operator and $\lambda>0$ globally dictates the strength of the
# regularization. Clearly, having one global regularization parameter $\lambda$ for the entire image is
# not optimal, as the image content can vary significantly across the image. Therefore, in this example,
# we aim to learn spatially adaptive regularization parameter maps $\Lambda_{\theta}$ from the input image to improve
# our TV-reconstruction. I.e., we are interested in estimating spatially adaptive regularization parameter maps by
#
# $\Lambda_{\theta}:=u_{\theta}(x_0)$
#
# with a convolutional neural network $u_{\theta}$ with trainable parameters $\theta$ from an input image $x_0$,
# and then to consider the weighted TV-minimization problem
#
# $\mathcal{F}_{\Lambda_{\theta}}(x) = \frac{1}{2}||Ax - y||_2^2 +  \| \Lambda_{\theta} \nabla x \|_1, \quad \quad (2)$
#
# where $\Lambda_{\theta}$ is voxel-wise strictly positive and locally regularizes the problem by
# differently weighting the gradient of the image.


# %% [markdown]
# ## The neural network for estimating the regularization parameter maps
# In this example, we use a simple convolutional neural network to estimate the regularization parameter maps.
# Obviously, more complex and sophisticated architectures can be employed as well. For example, in the work in
# [[Kofler et al, SIIMS 2023](https://epubs.siam.org/doi/abs/10.1137/23M1552486)],
# a U-Net [[Ronneberger et al, MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)] was used.
# The network used here corresponds to a simple block of convolutional layers with leaky ReLU activations and a
# final softplus activation to ensure that the regularization parameter maps are strictly positive.
# The network is defined in the following.

# %%
import torch


class ParameterMapNetwork2D(torch.nn.Module):
    r"""A simple network for estimating regularization parameter maps for TV-reconstruction."""

    def __init__(self, n_filters: int = 32) -> None:
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        n_filters
            Number of filters to be applied in the convolutional layers of the network.

        """
        super().__init__()
        self.cnn_block: torch.nn.Module = torch.nn.Sequential(
            *[
                torch.nn.InstanceNorm2d(1),
                torch.nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Softplus(beta=5.0),
            ]
        )
        # raw parameter t; softplus is used to "activate" it and make it strictly positive
        self.t = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""Apply the network to estimate regularization parameter maps.

        Parameters
        ----------
        image
            the image from which the regularization parameter maps should be estimated.
        """
        regularization_parameter_map = self.cnn_block(image)
        regularization_parameter_map = torch.nn.functional.softplus(self.t, beta=5.0) * regularization_parameter_map
        return regularization_parameter_map


# %% [markdown]
# ## The unrolled PDHG algorithm network
# We now construct the second block, i.e. network that (approximately) solves the TV-minimization problem with spatially
# adaptive regularization parameter maps given in (2) by unrolling a finite number of iterations of PDHG.

# %% [markdown]
# ```{note}
# To fully understand the mechanism of the network, we recommend to first have a look at the TV-example in
# <project:tv_minimization_reconstruction_pdhg.ipynb>, especially if you are not familiar with the PDHG algorithm
# or how to use it to solve the TV-minimization problem.
# ```

# %% [markdown]
# Put in simple words, the network takes the initial image, estimates the regularization
# parameter maps, and then sets up the TV-problem described in (2) within the "forward" of the network.
# The network then approximately solves the TV-problem using the PDHG algorithm
# and returns the reconstructed image.

# %%
import mrpro
from einops import rearrange


class AdaptiveTVNetwork2D(torch.nn.Module):
    r"""Unrolled primal dual hybrid gradient with spatially adaptive regularization parameter maps for TV.

    Solves the minimization problem

        :math:`\min_x \frac{1}{2}\| Ax - y\|_2^2 + \| \Lambda_{\theta} \nabla x\|_1`,
    where :math:`A` is the forward linear operator, :math:`\nabla` is the gradient operator,
    and :math:`\Lambda_{\theta}` is a strictly positive regularization parameter map that is estimated from
    an input image with a network :math:`u_{\theta}` with trainable parameters :math:`\theta`.

    N.B. The entire network sticks to the convention of MRpro, i.e. we work with images and k-space data
    of shape (other*, coils, z, y, x). However, because here showcase the method for 2D problems,
    some processing steps are necessary within the forward method. In particular, we restrict this example,
    to be used with z=1.
    """

    def __init__(
        self,
        recon_matrix: mrpro.data.SpatialDimension[int],
        encoding_matrix: mrpro.data.SpatialDimension[int],
        lambda_map_network: ParameterMapNetwork2D,
        n_iterations: int = 128,
    ):
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        recon_matrix
            image shape, i.e. the shape of the domain of the fourier operator.
        encoding_matrix
           k-space shape, i.e. the shape of the domain of the fourier operator
        lambda_map_network
            a network that predicts a regularization parameter map from the input image.
        n_iterations
            number of iterations for the unrolled primal dual hybrid gradient (PDHG) algorithm.

        """
        super().__init__()

        self.recon_matrix = recon_matrix
        self.encoding_matrix = encoding_matrix

        self.lambda_map_network = lambda_map_network
        self.n_iterations = n_iterations

        finite_differences_operator = mrpro.operators.FiniteDifferenceOp(dim=(-2, -1), mode='forward')
        self.gradient_operator = (
            mrpro.operators.RearrangeOp('grad batch ... -> batch grad ... ') @ finite_differences_operator
        )
        self.g = mrpro.operators.functionals.ZeroFunctional()

        # operator norm of the stacked operator K=[A, \nabla]^T
        stacked_operator_norm = 3.0  # analytically calculated
        self.primal_stepsize = self.dual_stepsize = 0.95 * (1.0 / stacked_operator_norm)

    def estimate_lambda_map(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate regularization parameter map from image.

        Parameters
        ----------
        image
            coil combined 2d image with shape `batch, coils=1, z=1, y x`
        """
        input_image = rearrange(image.abs(), '... 1 1 y x -> ... 1 y x')
        regularization_parameter_map = self.lambda_map_network(input_image).unsqueeze(-3).unsqueeze(-3)
        return regularization_parameter_map

    def forward(
        self,
        initial_image: torch.Tensor,
        kdata: torch.Tensor,
        forward_operator: mrpro.operators.LinearOperator,
        regularization_parameter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct image using an unrolled PDHG algorithm.

        Parameters
        ----------
        initial_image
            initial guess of the solution of the TV problem.
        kdata
            k-space data tensor of the considered problem.
        forward_operator
            forward operator that maps the image to the k-space data.
        regularization_parameter
            regularization parameter to be used in the TV-functional. If set to None,
            it is estimated by the lambda_map_network.
            (can also be a single scalar)

        Returns
        -------
            Image reconstructed by the PDGH algorithm to solve the weighted TV problem.
        """
        # if no regularization parameter map is provided, compute it with the network
        if regularization_parameter is None:
            regularization_parameter = self.estimate_lambda_map(initial_image)

        l2_norm_squared = 0.5 * mrpro.operators.functionals.L2NormSquared(target=kdata)
        l1_norm = mrpro.operators.functionals.L1NormViewAsReal(weight=regularization_parameter)
        f = mrpro.operators.ProximableFunctionalSeparableSum(l2_norm_squared, l1_norm)

        stacked_operator = mrpro.operators.LinearOperatorMatrix(((forward_operator,), (self.gradient_operator,)))

        (solution,) = mrpro.algorithms.optimizers.pdhg(
            f=f,
            g=self.g,
            operator=stacked_operator,
            initial_values=(initial_image,),
            max_iterations=self.n_iterations,
            primal_stepsize=self.primal_stepsize,
            dual_stepsize=self.dual_stepsize,
            tolerance=0.0,
        )
        return solution


# %% [markdown]
# ## Creating the training data
# In the following, we create some training data for the network. We use some images borrowed from the
# BrainWeb dataset [[Aubert-Broche et al, IEEE TMI 2006](https://ieeexplore.ieee.org/abstract/document/1717639)],
# which is a simulated MRI dataset and for which MRpro provides a simple interface to load the data, see
# `mrpro.phantoms.brainweb`.
# For simplicity, we here use some images that were generated by simulating the contrast using the
# inversion recovery signal model `mrpro.operators.models.InversionRecovery` in MRpro.
# In this notebook, we consider an ultra low field MR reconstruction problem, where the forward operator is given by
# $A:=S_I F$, where $F$ is a simple Cartesian Fourier operator with a single coil and $S_I$ corresponds to a mask
# only selecting the the central k-space region. Further, the present noise component is modeled to have a rather large
# variance. This means that in this setting, we tackle both the problem of super-resolution and denoising at once.


# %% [markdown]
# Download the data from Zenodo, create the datasets and dataloaders.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download and dataloading details"}
import tempfile
from pathlib import Path

import zenodo_get

dataset_tv_example = '15348115'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder_tv = Path(tmp.name)
zenodo_get.download(
    record=dataset_tv_example,
    retry_attempts=5,
    output_dir=data_folder_tv,
    file_glob=(
        'tv_model.pt',
        'training_images_part.pt',
        'validation_images_part.pt',
        'test_images_part.pt',
    ),
)  # r: retries

dataset_train = torch.utils.data.TensorDataset(
    torch.load(data_folder_tv / 'training_images_part.pt').unsqueeze(-3).unsqueeze(-3)
)
dataset_validation = torch.utils.data.TensorDataset(
    torch.load(data_folder_tv / 'validation_images_part.pt').unsqueeze(-3).unsqueeze(-3)
)
dataset_test = torch.utils.data.TensorDataset(
    torch.load(data_folder_tv / 'test_images_part.pt').unsqueeze(-3).unsqueeze(-3)
)

dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=1)
dataloader_validation = torch.utils.data.DataLoader(dataset_validation, shuffle=False, batch_size=1)
dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=1)

# %% [markdown]
# We define some auxiliary functions that we will need for retrospectively simulating undersampled
# and noisy k-space data.


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show data preparation details"}
def normalize_kspace_data_and_image(
    kdata: torch.Tensor,
    target_image: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize k-space data and (possibly) target image."""
    factor = 1.0 / kdata.abs().std()
    kdata *= factor
    if target_image is not None:
        target_image *= factor
    return kdata, target_image


def prepare_data(
    target_image: torch.Tensor, super_resolution_factor: float, noise_variance: float, seed: int = 0
) -> tuple[torch.Tensor, mrpro.operators.LinearOperator, torch.Tensor, torch.Tensor]:
    """Prepare data for training by simulating k-space data.

    Parameters
    ----------
    target_image : torch.Tensor
        Target image to be used for simulation.
    super_resolution_factor : float
        Factor determining by how much the number of pixels is increased (i.e. how much smaller
        each pixel will be).
    noise_variance : float
        Variance of the Gaussian noise to be added.
    seed : float
        seed of the random number generator.

    Returns
    -------
    tuple
        A tuple containing k-space data, forward operator adjoint reconstruction,
        possibly the pseudo-inverse solution (if n_coils>1) and the target image.
    """
    # randomly choose trajectories to define the fourier operator
    ny, nx = target_image.shape[-2:]
    recon_matrix = mrpro.data.SpatialDimension(z=1, y=ny, x=nx)
    encoding_matrix = mrpro.data.SpatialDimension(z=1, y=ny, x=nx)

    n_k1, n_k0 = ny // super_resolution_factor, nx // super_resolution_factor

    traj = mrpro.data.traj_calculators.KTrajectoryCartesian()(
        n_k0=int(n_k0),
        k0_center=int(n_k0 // 2),
        k1_idx=torch.arange(-n_k1 // 2, n_k1 // 2)[..., None, None, :, None],
        k1_center=0,
        k2_idx=torch.tensor(0),
        k2_center=0,
    )

    fourier_operator = mrpro.operators.FourierOp(
        traj=traj,
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
    )

    (kdata,) = fourier_operator(target_image)

    rng = torch.Generator().manual_seed(seed)
    kdata = kdata + noise_variance * kdata.std() * torch.randn(
        kdata.shape, dtype=kdata.dtype, device=kdata.device, generator=rng
    )

    kdata, target_image = normalize_kspace_data_and_image(kdata, target_image)  # type: ignore[assignment]

    (adjoint_recon,) = fourier_operator.H(kdata)

    return kdata, fourier_operator, adjoint_recon, target_image


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
# Define some functions for plotting images.
import matplotlib.pyplot as plt
import torch


def show_images(
    *images: torch.Tensor,
    titles: list[str] | None = None,
    cmap: str = 'grey',
    clim: tuple[float, float] | None = None,
    rotation_k: int = 0,
    colorbar: bool = False,
    # show_mse: bool = False,
    show_ssim: bool = False,
) -> None:
    """Plot images."""
    if clim is None:
        clim = (0.0, 1.0)
    n_images = len(images)
    single_figsize = 2.5
    fig, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * single_figsize, single_figsize))

    image_reference = images[-1].cpu() if images[-1].is_cuda else images[-1]
    if rotation_k != 0:
        image_reference = image_reference.rot90(k=rotation_k, dims=(-2, -1))
    for i in range(n_images):
        image = images[i].cpu() if images[i].is_cuda else images[i]
        if rotation_k != 0:
            image = image.rot90(k=rotation_k, dims=(-2, -1))
        im = axes[0, i].imshow(image.abs().squeeze(), cmap=cmap, clim=clim)
        if titles:
            axes[0, i].set_title(titles[i], fontsize=12)
        if colorbar and i == n_images - 1:
            fig.colorbar(im, ax=axes[0, i])

        if show_ssim and i != n_images - 1:
            ssim = mrpro.operators.functionals.SSIM(target=image_reference)
            (ssim_value,) = ssim(image)
            axes[0, i].text(
                0.5,
                0.12,
                f'SSIM: {ssim_value.item():.2f}',
                color='white',
                fontsize=12,
                horizontalalignment='center',
                verticalalignment='top',
                transform=axes[0, i].transAxes,
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 1,
                },
            )

        axes[0, i].set_axis_off()
    fig.subplots_adjust(wspace=0.03)
    plt.show()


# %% [markdown]
# Let us have a look at one example of the BrainWeb images. From a target image, we can create the k-space
# data by applying the forward operator and adding Gaussian noise. Then, we compute the adjoint reconstruction.
# The function `prepare_data` (defined further above) does all of this for us.

# %%
# select sample id 6 (you can change this to any other id, but sample 6 is a good one,
# as it has some nice contrast)
import itertools

sample_id = 6
(image,) = next(itertools.islice(dataloader_test, sample_id, sample_id + 1))

super_resolution_factor = 2.0
noise_variance = 0.4

kdata, fourier_operator, adjoint_recon, image = prepare_data(
    image, super_resolution_factor=super_resolution_factor, noise_variance=noise_variance
)

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show training details"}
images_list = [adjoint_recon, image]
titles = ['Adjoint', 'Ground Truth']
show_images(
    *images_list,
    titles=titles,
    show_ssim=True,
    rotation_k=-1,
    clim=(0, 0.8 * image.abs().max().item()),
)

# %% [markdown]
# ## Network training

# %% [markdown]
# We now define the unrolled PDHG network by instantiating the `AdaptiveTVNetwork2D` class.

# %%
torch.manual_seed(2025)
lambda_map_network = ParameterMapNetwork2D(n_filters=128)
n_iterations = 128

recon_matrix = mrpro.data.SpatialDimension(z=1, y=image.shape[-2], x=image.shape[-1])
encoding_matrix = mrpro.data.SpatialDimension(z=1, y=image.shape[-2], x=image.shape[-1])

adaptive_tv_network = AdaptiveTVNetwork2D(
    recon_matrix=recon_matrix,
    encoding_matrix=encoding_matrix,
    lambda_map_network=lambda_map_network,
    n_iterations=n_iterations,
)


# %% [markdown]
# Let us have a look at the regularization parameter map and the corresponding TV-reconstruction
# prior to network training.

# %%
if torch.cuda.is_available():
    adaptive_tv_network = adaptive_tv_network.cuda()
    adjoint_recon = adjoint_recon.cuda()
    kdata = kdata.cuda()
    fourier_operator = fourier_operator.cuda()

regularization_parameter_map_before_training = adaptive_tv_network.estimate_lambda_map(adjoint_recon).detach()

pdhg_recon_before_training = (
    adaptive_tv_network(adjoint_recon, kdata, fourier_operator, regularization_parameter_map_before_training)
    .detach()
    .cpu()
)


# %% [markdown]
# Let us see how we could train the unrolled PDHG network. We set up the optimizer with an appropriate learning rate,
# number of epochs and write a simple training loop, in which we present the network input-target pairs and update
# the network's parameters by gradient-descent.
# Training neural networks, especially those based on algorithm unrolling, is time consuming. We therefore
# only perform a few weight updates here to showcase that the MSE is reduced during training. If you want to train the
# network on some on other data, you can use the following code snippet as a template and increase the number of images
# used for training as well as the number of epochs.

# %%
optimizer = torch.optim.Adam(
    [
        {'params': adaptive_tv_network.lambda_map_network.cnn_block.parameters(), 'lr': 1e-5},
        {'params': [adaptive_tv_network.lambda_map_network.t], 'lr': 1e-2},
    ],
)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

import copy
import datetime
import itertools
from time import time

from tqdm import tqdm

n_epochs = 4
validation_loss_values = []
best_model = None

n_training = 16
n_validation = 8

time_start = time()
outer_bar = tqdm(range(n_epochs), desc='Epochs', disable=False)
for epoch in outer_bar:
    for sample_num, (image_,) in enumerate(
        itertools.islice(dataloader_train, n_training)
    ):  # here, we only use 16 images for training
        optimizer.zero_grad()

        kdata_, fourier_operator_, adjoint_recon_, image_ = prepare_data(
            image_,
            super_resolution_factor=super_resolution_factor,
            noise_variance=noise_variance,
            seed=sample_num * epoch,
        )

        if torch.cuda.is_available():
            kdata_ = kdata_.cuda()
            adjoint_recon_ = adjoint_recon_.cuda()
            fourier_operator_ = fourier_operator_.cuda()
            image_ = image_.cuda()

        pdhg_recon_ = adaptive_tv_network(adjoint_recon_, kdata_, fourier_operator_)

        loss = torch.nn.functional.mse_loss(torch.view_as_real(pdhg_recon_), torch.view_as_real(image_))

        loss.backward()
        optimizer.step()

    lr_scheduler.step()

    running_loss = 0.0
    for sample_num, (image_,) in enumerate(
        itertools.islice(dataloader_validation, n_validation)
    ):  # here, we only use 4 images for validation
        kdata_, fourier_operator_, adjoint_recon_, image_ = prepare_data(
            image_,
            super_resolution_factor=super_resolution_factor,
            noise_variance=noise_variance,
            seed=sample_num,
        )

        if torch.cuda.is_available():
            kdata_ = kdata_.cuda()
            adjoint_recon_ = adjoint_recon_.cuda()
            fourier_operator_ = fourier_operator_.cuda()
            image_ = image_.cuda()

        with torch.no_grad():
            pdhg_recon_ = adaptive_tv_network(adjoint_recon_, kdata_, fourier_operator_)
            loss = torch.nn.functional.mse_loss(torch.view_as_real(pdhg_recon_), torch.view_as_real(image_))

        running_loss += loss.item()
    loss_validation = running_loss / len(dataset_validation)
    validation_loss_values.append(loss_validation)
    outer_bar.set_postfix(val_loss=f'{loss_validation:.8f}')
    if validation_loss_values[-1] <= min(validation_loss_values):
        # store the weights if the validation loss is the smallest; this is
        # to avoid possible overfitting here, since we are only using very few
        # images
        best_model = copy.deepcopy(adaptive_tv_network.state_dict())

time_end = time() - time_start
print(f'training time: {datetime.timedelta(seconds=time_end)}')

fig, ax = plt.subplots()
ax.plot(range(1, n_epochs + 1), validation_loss_values, label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Mean Squared Error')
ax.legend()

# %% [markdown]
# Voilà! As you we can see, the network seems to learn something. A proper exhaustive training would
# take too long. Therefore, we here also provide a pre-trained model that we
# can load and use for the reconstruction.

# %%
adaptive_tv_network.load_state_dict(torch.load(data_folder_tv / 'tv_model.pt', map_location='cpu'))

regularization_parameter_map_trained = adaptive_tv_network.estimate_lambda_map(adjoint_recon).detach()
pdhg_recon_regularization_parameter_trained = (
    (adaptive_tv_network(adjoint_recon, kdata, fourier_operator, regularization_parameter_map_trained).detach().cpu())
    .detach()
    .cpu()
)

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
show_images(
    regularization_parameter_map_before_training[0, 0],
    regularization_parameter_map_trained[0, 0].cpu(),
    titles=[
        'Regularization  Map \n (Before Training)',
        'Regularization  Map \n (After Training)',
    ],
    cmap='inferno',
    rotation_k=-1,
    clim=(0.0, 0.5 * regularization_parameter_map_trained.abs().max().item()),
    colorbar=True,
)

# %% [markdown]
# Also, let us check that the learned regularization parameter maps improved the reconstructions
# over the ones obtained without training.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
show_images(
    adjoint_recon[0, 0],
    pdhg_recon_before_training[0, 0],
    pdhg_recon_regularization_parameter_trained[0, 0].cpu(),
    image[0, 0],
    titles=[
        'Input Image',
        'Spatially Adaptive TV \n (Before Training)',
        'Spatially Adaptive TV \n (After Training)',
        'Ground Truth',
    ],
    cmap='grey',
    show_ssim=True,
    rotation_k=-1,
    clim=(0.0, 0.9 * image.abs().max().item()),
    colorbar=False,
)

# Additionally, let us also perform a quick line search to see what the best possible TV-reconstruction using a
# scalar regularization parameter would be. Note that in practice, you would obviously not be able
# to obtain this reconstruction since the target image is not available. However, the comparison
# is useful to assess how much improvement one can expect to obtain when employing
# spatially adaptive regularization parameter maps.


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show line search details"}
def line_search(
    regularization_parameters: torch.Tensor,
    initial_image: torch.Tensor,
    kdata: torch.Tensor,
    forward_operator: mrpro.operators.LinearOperator,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform a line search to pick the best scalar regularization parameter for TV."""
    reconstructions_list = [
        adaptive_tv_network(initial_image, kdata, forward_operator, regularization_parameter)
        for regularization_parameter in regularization_parameters
    ]
    mse_values = torch.tensor(
        [
            torch.nn.functional.mse_loss(torch.view_as_real(recon), torch.view_as_real(target))
            for recon in reconstructions_list
        ]
    )

    return mse_values, reconstructions_list[torch.argmin(torch.tensor(mse_values))]


regularization_parameters = torch.linspace(1e-1, 3e-1, 16)

mse_values, pdhg_recon_best_scalar = line_search(
    regularization_parameters,
    adjoint_recon,
    kdata,
    fourier_operator,
    (image).cuda() if torch.cuda.is_available() else image,
)

from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots()
ax.plot(regularization_parameters, mse_values.cpu())
ax.set_xlabel(r'Scalar Regularization Parameter Value $\lambda$', fontsize=12)
ax.set_ylabel(r'MSE (TV($\lambda$),Target)', fontsize=12)
ax.vlines(
    x=regularization_parameters[torch.argmin(mse_values)].item(),
    ymin=mse_values.min().item(),
    ymax=mse_values.max().item(),
    colors='red',
    ls=':',
    label=r'Best scalar $\lambda>0$',
)

ax.legend()
ax.tick_params(axis='x', rotation=45)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
plt.show()

# %% [markdown]
# Below, you can see the adjoint reconstruction, the PDHG reconstruction with the best scalar regularization
# parameter, the PDHG reconstruction with the spatially adaptive as well as the target image.


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
show_images(
    adjoint_recon,
    pdhg_recon_best_scalar,
    pdhg_recon_regularization_parameter_trained,
    image,
    titles=[
        'Adjoint',
        'TV with\n' + r'Best Scalar $\lambda$',
        'TV with\n' + r'Learned $\boldsymbol{\Lambda}_{\theta}$-Map',
        'Ground Truth',
    ],
    show_ssim=True,
    rotation_k=-1,
    clim=(0.0, 0.9 * image.abs().max().item()),
)
# %% [markdown]
# ## Application to in-vivo data
# Finally, let us download some ultra low field scanner data, apply the pre-trained network and
# visualize the results to see if the network generalizes well to in-vivo data.

pname = '/echo/kofler01/lowfield_data/'
# fname = 'pd_data_2d.mrd'
fname = 't2_data_2d.mrd'

kdata_scanner = mrpro.data.KData.from_file(
    pname + fname,
    mrpro.data.traj_calculators.KTrajectoryCartesian(),
)

kdata_scanner.data, _ = normalize_kspace_data_and_image(kdata_scanner.data, None)

direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_scanner)
low_resolution_adjoint_recon = direct_reconstruction(kdata_scanner)

# %%
ny_low_resolution, nx_low_resolution = kdata_scanner.data.shape[-2], kdata_scanner.data.shape[-1]
nz, ny_target, nx_target = (
    1,
    int(super_resolution_factor * ny_low_resolution),
    int(super_resolution_factor * nx_low_resolution),
)  # target resolution
n_k1, n_k0 = ny_low_resolution, nx_low_resolution

recon_matrix = encoding_matrix = mrpro.data.SpatialDimension(z=nz, y=ny_target, x=nx_target)

traj = mrpro.data.traj_calculators.KTrajectoryCartesian()(
    n_k0=int(n_k0),
    k0_center=int(n_k0 // 2),
    k1_idx=torch.arange(-n_k1 // 2, n_k1 // 2)[..., None, None, :, None],
    k1_center=0,
    k2_idx=torch.tensor(0),
    k2_center=0,
)

fourier_operator_scanner = mrpro.operators.FourierOp(
    recon_matrix=recon_matrix,
    encoding_matrix=encoding_matrix,
    traj=traj,
)

(super_resolution_adjoint_reco,) = fourier_operator_scanner.H(kdata_scanner.data)

# %%
if torch.cuda.is_available():
    super_resolution_adjoint_reco = super_resolution_adjoint_reco.cuda()
    kdata_scanner = kdata_scanner.cuda()
    fourier_operator_scanner = fourier_operator_scanner.cuda()

regularization_parameter_map_scanner_data = adaptive_tv_network.estimate_lambda_map(
    super_resolution_adjoint_reco
).detach()

pdhg_recon_scanner_data = (
    adaptive_tv_network(
        super_resolution_adjoint_reco,
        kdata_scanner.data,
        fourier_operator_scanner,
        regularization_parameter_map_scanner_data,
    )
    .detach()
    .cpu()
)

show_images(
    regularization_parameter_map_scanner_data[0, 0],
    titles=['Predicted Regularization \n Parameter Map'],
    cmap='inferno',
    rotation_k=-1,
    clim=(0.0, 0.4 * regularization_parameter_map_scanner_data.abs().max().item()),
    colorbar=True,
)

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
show_images(
    1 / super_resolution_factor * low_resolution_adjoint_recon.data.squeeze(),
    pdhg_recon_scanner_data.squeeze(),
    titles=[
        'Ultra-Low Field\n Adjoint Reconstruction',
        'Ultra-Low Field\n' + r'TV with $\boldsymbol{\Lambda}_{\theta}$-Map',
    ],
    show_ssim=False,
    rotation_k=-1,
    clim=(0.0, 0.6 * super_resolution_adjoint_reco.abs().max().item()),
)

# %%
