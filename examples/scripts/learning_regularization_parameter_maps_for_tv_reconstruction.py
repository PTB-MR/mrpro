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
# estimate spatially adaptive regularization parameter maps from an input image.

# %% [markdown]
# ## The method
# In the TV-example, (see <project:tv_minimization_reconstruction_pdhg.ipynb>), you can see how to employ the primal
# dual hybrid gradient (PDHG - `mrpro.algorithms.optimizers.pdhg`) method
# [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)]
# to solve the TV-minimization problem. There, for data acquired according to the usual model
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

    def __init__(self, n_filters: int = 16) -> None:
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        n_filters
            Number of filters to be applied in the convolutional layers of the network.

        """
        super().__init__()
        self.beta_softplus = 10.0
        self.cnn_block: torch.nn.Module = torch.nn.Sequential(
            *[
                torch.nn.InstanceNorm2d(1, affine=False, track_running_stats=False),
                torch.nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Softplus(beta=self.beta_softplus),
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
        regularization_parameter_map = (
            torch.nn.functional.softplus(self.t, beta=self.beta_softplus) * regularization_parameter_map
        )
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
        gradient_operator = (
            mrpro.operators.RearrangeOp('grad batch ... -> batch grad ... ') @ finite_differences_operator
        )
        self.gradient_operator = gradient_operator

        self.g = mrpro.operators.functionals.ZeroFunctional()

        # operator norm of the stacked operator K=[A, \nabla]^T
        stacked_operator_norm = 3.0  # analytically calculated
        self.primal_stepsize = self.dual_stepsize = 0.95 * (1.0 / stacked_operator_norm)

    def estimate_lambda_map(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate regularization parameter map from image."""
        # The initial image is a coil-combined image. Since we only consider 2D problems
        # here, z can be assumed to be always 1.
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

        f1 = 0.5 * mrpro.operators.functionals.L2NormSquared(target=kdata)
        f2 = mrpro.operators.functionals.L1NormViewAsReal(weight=regularization_parameter)
        f = mrpro.operators.ProximableFunctionalSeparableSum(f1, f2)

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
# which is a simulated MRI dataset and for which MRpro provides a simple interface to load the data.
# For simplicity, we here use some images that were generated by simulating the contrast using the
# inversion recovery signl model in MRpro.
# First, we start by defining some auxiliary functions that we will need for retrospectively simulating undersampled
# and noisy k-space data.

# %%


def add_gaussian_noise(kdata: torch.Tensor, noise_variance: float, seed: int) -> torch.Tensor:
    """Corrupt given k-space data with additive Gaussian noise."""
    rng = torch.Generator().manual_seed(seed)
    kdata = kdata + noise_variance * kdata.std() * torch.randn(
        kdata.shape, dtype=kdata.dtype, device=kdata.device, generator=rng
    )

    return kdata


def normalize_kspace_data_and_image(
    kdata: torch.Tensor,
    target_image: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize k-space data and (possibly) target image."""
    factor = 1 / kdata.abs().max()
    kdata *= factor
    if target_image is not None:
        target_image *= factor
    return kdata, target_image


# %% [markdown]
# Further, we define a torch dataset that will be used for training the network.
# %%


class Cartesian2D(torch.utils.data.Dataset):
    r"""Dataset for 2D problems."""

    def __init__(self, images: torch.Tensor) -> None:
        """Parameters

        ----------
        images
            images to be used in the dataset.
            Should have the shape (others, coils, z, y, x) with z=1.
        """
        self.images = images

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """Return the image indexed by idx."""
        image = self.images[idx]

        return image


# %% [markdown]
# We also define a function that prepares the data for training. It simulates the k-space data by applying the
# forward operator to the image and adds Gaussian noise. The function also computes the adjoint reconstruction
# and the pseudo-inverse solution, which can be used as input image to estimate the regularization parameter maps.


# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show data preparation details"}


def prepare_data(
    target_image: torch.Tensor, n_coils: int, acceleration_factor: float, noise_variance: float, seed: int = 0
) -> tuple[torch.Tensor, mrpro.operators.LinearOperator, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Prepare data for training by simulating k-space data.

    Parameters
    ----------
    target_image : torch.Tensor
        Target image to be used for simulation.
    n_coils : int
        Number of coils for the simulation.
    acceleration_factor : float
        Acceleration factor for undersampling.
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

    traj = mrpro.data.traj_calculators.KTrajectoryCartesian().gaussian_variable_density(
        encoding_matrix=encoding_matrix, acceleration=acceleration_factor, fwhm_ratio=0.2, seed=seed
    )

    fourier_operator = mrpro.operators.FourierOp(
        traj=traj,
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
    )

    # generate simualated coil sensitivity maps
    from mrpro.phantoms.coils import birdcage_2d

    if n_coils > 1:
        csm = birdcage_2d(n_coils, recon_matrix, relative_radius=0.8)
        csm_operator = mrpro.operators.SensitivityOp(csm)
        forward_operator = fourier_operator @ csm_operator
    else:
        forward_operator = fourier_operator

    (kdata,) = forward_operator(target_image)
    kdata = add_gaussian_noise(kdata, noise_variance=noise_variance, seed=seed)

    kdata, target_image = normalize_kspace_data_and_image(kdata, target_image)  # type: ignore[assignment]

    (adjoint_recon,) = forward_operator.H(kdata)

    if n_coils > 1:
        # compute an approximation of the pseudo-inverse solution
        # Note that this in general only makes sense if the forward operator can be injective,
        # i.e. we have multiple coils with acceleration factor < n_coils
        (pseudo_inverse_solution,) = mrpro.algorithms.optimizers.cg(
            forward_operator.gram, right_hand_side=adjoint_recon, initial_value=adjoint_recon, max_iterations=16
        )
        return kdata, forward_operator, adjoint_recon, pseudo_inverse_solution, target_image
    else:
        pseudo_inverse_solution = None
    return kdata, forward_operator, adjoint_recon, pseudo_inverse_solution, target_image


# %% [markdown]
# Download the data from Zenodo, create the datasets and dataloaders and define
# some auxiliary functions for displaying the images.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download and dataloading details"}
import tempfile
from pathlib import Path

import zenodo_get

dataset = '15407890'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder_tv = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder_tv])  # r: retries

n_images = 28
images = torch.load(data_folder_tv / 'brainweb_data.pt')[:n_images, ...]
images = images.unsqueeze(-3).unsqueeze(-3)
dataset_train = Cartesian2D(images[:20, ...])
dataset_validation = Cartesian2D(images[20:, ...])

dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=1)
dataloader_validation = torch.utils.data.DataLoader(dataset_validation, shuffle=False, batch_size=1)

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
# Define some functions for plotting images.
import matplotlib.pyplot as plt
import torch


def show_images(
    *images: torch.Tensor,
    titles: list[str] | None = None,
    cmap: str = 'grey',
    clim: tuple[float, float] | None = None,
    rotate: bool = True,
    rotatation_k: int = -2,
    colorbar: bool = False,
    show_mse: bool = False,
) -> None:
    """Plot images."""
    if clim is None:
        clim = (0.0, 1.0)
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 3, 3))

    image_reference = images[-1].cpu() if images[-1].is_cuda else images[-1]
    if rotate:
        image_reference = image_reference.rot90(k=rotatation_k, dims=(-2, -1))
    for i in range(n_images):
        image = images[i].cpu() if images[i].is_cuda else images[i]
        if rotate:
            image = image.rot90(k=rotatation_k, dims=(-2, -1))
        im = axes[0, i].imshow(image.abs(), cmap=cmap, clim=clim)
        if titles:
            axes[0, i].set_title(titles[i])
        if colorbar:
            fig.colorbar(im, ax=axes[0, i])
        if show_mse and i != n_images - 1:
            mse = torch.nn.functional.mse_loss(torch.view_as_real(image), torch.view_as_real(image_reference)).item()
            axes[0, i].text(
                10, 25, f'MSE={mse:.2e}', fontsize=12, color='yellow', bbox={'facecolor': 'grey', 'boxstyle': 'round'}
            )
        axes[0, i].set_axis_off()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# Let us have a look at one example of the BrainWeb images.

# %%

image_ = next(iter(dataloader_validation))[0]

n_coils = 8
acceleration_factor = 4.0
noise_variance = 0.05

kdata_, forward_operator_, adjoint_recon_, pseudo_inverse_solution_, image_ = prepare_data(
    image_, n_coils=n_coils, acceleration_factor=acceleration_factor, noise_variance=noise_variance
)
if n_coils > 1:
    assert pseudo_inverse_solution_ is not None
    images_list = [adjoint_recon_.squeeze(), pseudo_inverse_solution_.squeeze(), image_.squeeze()]
    titles = ['Adjoint', 'Pseudo-Inverse', 'Target']
else:
    images_list = [adjoint_recon_.squeeze(), image_.squeeze()]
    titles = ['Adjoint', 'Target']
show_images(
    *images_list,
    titles=titles,
    show_mse=True,
    clim=(0, 0.8 * image_.abs().max().item()),
)

# %% [markdown]
# ## Network training

# %% [markdown]
# We now define the unrolled PDHG network by instantiating the `AdaptiveTVNetwork2D` class.

# %%
torch.manual_seed(2025)
lambda_map_network = ParameterMapNetwork2D(n_filters=32)
n_iterations = 128

recon_matrix = mrpro.data.SpatialDimension(z=1, y=images.shape[-2], x=images.shape[-1])
encoding_matrix = mrpro.data.SpatialDimension(z=1, y=images.shape[-2], x=images.shape[-1])

adaptive_tv_network = AdaptiveTVNetwork2D(
    recon_matrix=recon_matrix,
    encoding_matrix=encoding_matrix,
    lambda_map_network=lambda_map_network,
    n_iterations=n_iterations,
)

if n_coils > 1:
    assert pseudo_inverse_solution_ is not None
    input_image_ = pseudo_inverse_solution_
else:
    input_image_ = adjoint_recon_

if torch.cuda.is_available():
    adaptive_tv_network = adaptive_tv_network.cuda()
    input_image_ = input_image_.cuda()
    kdata_ = kdata_.cuda()
    forward_operator_ = forward_operator_.cuda()

with torch.no_grad():
    regularization_parameter_map_init = adaptive_tv_network.estimate_lambda_map(input_image_)


# %% [markdown]
# Let us now train the unrolled PDHG network. We set up the optimizer with an appropriate learning rate, choose a
# number of epochs and write a simple training loop, in which we present the network input-target pairs and update
# the network's parameters by gradient-descent.
# To obtain a somewhat decent estimate of the network parameters, training should take approximately 5-6 minutes
# on a GPU. If you have GPU and you need a break, start the training now, grab yourself a coffee and come back later
# to see your network trained from scratch.
# If you do not have a GPU, we provide a pre-trained model, which you can load and directly apply to the data.

# %%
# cnn_block = cast(torch.nn.Module, adaptive_tv_network.lambda_map_network.cnn_block)
# cnn_block =
optimizer = torch.optim.Adam(
    [
        {'params': adaptive_tv_network.lambda_map_network.cnn_block.parameters(), 'lr': 1e-4},
        {'params': [adaptive_tv_network.lambda_map_network.t], 'lr': 1e-2},
    ],
    weight_decay=1e-8,
)

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

import copy
import datetime
from time import time

from tqdm import tqdm

n_epochs = 10
validation_loss_values = []
best_model = None

if not torch.cuda.is_available():
    # without a gpu, skip training. instead, load the pre-trained model downloaded from zenodo
    adaptive_tv_network.load_state_dict(torch.load(data_folder_tv / 'tv_model.pt', map_location='cpu'))
else:
    time_start = time()
    outer_bar = tqdm(range(n_epochs), desc='Epochs', disable=False)
    for epoch in outer_bar:
        for sample_num, image_ in enumerate(dataloader_train):
            optimizer.zero_grad()

            kdata, forward_operator_undersampled, adjoint_recon, pseudo_inverse_solution, image = prepare_data(
                image_,
                n_coils=n_coils,
                acceleration_factor=acceleration_factor,
                noise_variance=noise_variance,
                seed=sample_num * epoch,
            )

            initial_image = adjoint_recon if pseudo_inverse_solution is None else pseudo_inverse_solution

            if torch.cuda.is_available():
                kdata = kdata.cuda()
                initial_image = initial_image.cuda()
                forward_operator_undersampled = forward_operator_undersampled.cuda()
                image = image.cuda()

            pdhg_recon = adaptive_tv_network(initial_image, kdata, forward_operator_undersampled)

            loss = torch.nn.functional.mse_loss(torch.view_as_real(pdhg_recon), torch.view_as_real(image))

            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        running_loss = 0.0
        for sample_num, image_ in enumerate(dataloader_validation):
            kdata, forward_operator_undersampled, adjoint_recon, pseudo_inverse_solution, image = prepare_data(
                image_,
                n_coils=n_coils,
                acceleration_factor=acceleration_factor,
                noise_variance=noise_variance,
                seed=sample_num,
            )

            initial_image = adjoint_recon if pseudo_inverse_solution is None else pseudo_inverse_solution

            if torch.cuda.is_available():
                kdata = kdata.cuda()
                initial_image = initial_image.cuda()
                forward_operator_undersampled = forward_operator_undersampled.cuda()
                image = image.cuda()

            with torch.no_grad():
                # pdhg_recon = adaptive_tv_network(pseudo_inverse_solution, csm, kdata, traj)
                pdhg_recon = adaptive_tv_network(initial_image, kdata, forward_operator_undersampled)
                loss = torch.nn.functional.mse_loss(torch.view_as_real(pdhg_recon), torch.view_as_real(image))

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

    assert best_model is not None
    adaptive_tv_network.load_state_dict(best_model)

    fig, ax = plt.subplots()
    ax.plot(range(1, n_epochs + 1), validation_loss_values, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()


# %% [markdown]
# Nice! We have now trained our network for estimating the regularization parameter maps. Let us check if the obtained
# maps show interesting features by applying it to the previous image.

# %%
with torch.no_grad():
    regularization_parameter_map_trained = adaptive_tv_network.estimate_lambda_map(input_image_)

    pdhg_recon_regularization_parameter_trained_ = adaptive_tv_network(
        input_image_, kdata_, forward_operator_, regularization_parameter_map_trained
    )

show_images(
    regularization_parameter_map_init[0, 0].squeeze(),
    regularization_parameter_map_trained[0, 0].squeeze().cpu(),
    titles=[
        'Reg. Parameter Map \n (Before Training)',
        'Reg. Parameter Map \n (After Training)',
    ],
    cmap='inferno',
    clim=(0.0, regularization_parameter_map_trained.abs().max().item()),
    colorbar=True,
)


# %% [markdown]
# ## Application to in-vivo data
# Let us now download some real measured scanner data and apply our trained network. We will use the same dataset of the
# Cartesian reconstruction example <project:cartesian_reconstruction.ipynb>.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Get the raw Carestian MR data from zenodo
import tempfile
from pathlib import Path

import zenodo_get

dataset = '14173489'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder_cart_data = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder_cart_data])  # r: retries

kdata_cartesian = mrpro.data.KData.from_file(
    data_folder_cart_data / 'cart_t1.mrd',
    mrpro.data.traj_calculators.KTrajectoryCartesian(),
)

# %% [markdown]
# From the measured data, we select a subset of the sampled k-space lines to create an undersampled k-space
# data. Then, we estimate the coil sensitivity maps using the Walsh method implemented in MRpro and obtain the initial
# reconstruction by approximately solving the normal equations using conjugate gradient method.

# %%
nz, ny, nx = kdata_cartesian.data.shape[-3:]
encoding_matrix_cartesian = mrpro.data.SpatialDimension(z=nz, y=ny, x=nx)
n_k1 = kdata_cartesian.data.shape[-2]
k1_center = n_k1 // 2

ktraj_undersampled = mrpro.data.traj_calculators.KTrajectoryCartesian().gaussian_variable_density(
    encoding_matrix=encoding_matrix_cartesian, acceleration=acceleration_factor, fwhm_ratio=0.2, seed=2025
)

k1_idx = ktraj_undersampled.ky.squeeze() + k1_center
kdata_undersampled = kdata_cartesian.data[..., k1_idx.to(torch.int), :]

fourier_operator_undersampled = mrpro.operators.FourierOp(
    traj=ktraj_undersampled,
    recon_matrix=kdata_cartesian.header.recon_matrix,
    encoding_matrix=kdata_cartesian.header.encoding_matrix,
)

kdata_undersampled_, _ = normalize_kspace_data_and_image(kdata_undersampled, None)
assert isinstance(kdata_undersampled_, torch.Tensor)
kdata_undersampled = kdata_undersampled_


(coil_images,) = fourier_operator_undersampled.H(kdata_undersampled)

csm_undersampled = mrpro.algorithms.csm.walsh(coil_images.squeeze(0), smoothing_width=5).unsqueeze(0)

forward_operator_undersampled = fourier_operator_undersampled @ mrpro.operators.SensitivityOp(csm_undersampled)
(adjoint_recon_undersampled,) = forward_operator_undersampled.H(kdata_undersampled)

(pseudo_inverse_solution_undersampled,) = mrpro.algorithms.optimizers.cg(
    forward_operator_undersampled.gram,
    right_hand_side=adjoint_recon_undersampled,
    initial_value=adjoint_recon_undersampled,
    max_iterations=16,
)

if torch.cuda.is_available():
    pseudo_inverse_solution_undersampled = pseudo_inverse_solution_undersampled.cuda()
    forward_operator_undersampled = forward_operator_undersampled.cuda()
    kdata_undersampled = kdata_undersampled.cuda()

# %% [markdown]
# Finally, let us apply the trained network to the undersampled data. We first estimate the regularization parameter map
# from the pseudo-inverse solution and then apply the unrolled PDHG network to obtain the final reconstruction.

# %%
adaptive_tv_network.recon_matrix = kdata_cartesian.header.recon_matrix
adaptive_tv_network.encoding_matrix = kdata_cartesian.header.encoding_matrix

with torch.no_grad():
    regularization_parameter_map_undersampled_data = adaptive_tv_network.estimate_lambda_map(
        pseudo_inverse_solution_undersampled
    )

    pdhg_recon_regularization_parameter_trained_ = adaptive_tv_network(
        pseudo_inverse_solution_undersampled,
        kdata_undersampled,
        forward_operator_undersampled,
        regularization_parameter_map_undersampled_data,
    )

# %% [markdown]
# Let us have a look at the reconstruced image as well as as the estimate regularization parameter
# map. Also, let us compare the reconstructed image to the one reconstructed from the fully-sampled k-space data using
# the iterative SENSE reconstruction class.

# %%
# Create DirectReconstruction object from KData object. Also, scale the k-space data to better match the
# intensities of the undersampled k-space data.

fourier_operator_full = mrpro.operators.FourierOp(
    traj=kdata_cartesian.traj,
    recon_matrix=kdata_cartesian.header.recon_matrix,
    encoding_matrix=kdata_cartesian.header.encoding_matrix,
)

kdata_cartesian_data_normalized, _ = normalize_kspace_data_and_image(kdata_cartesian.data, None)
assert isinstance(kdata_cartesian_data_normalized, torch.Tensor)
kdata_cartesian.data = kdata_cartesian_data_normalized
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_cartesian)
image_fully_sampled = iterative_sense_reconstruction(kdata_cartesian)


# %% [markdown]
# Additionally, we also perform a quick line search to see what the best possible TV-reconstruction using a
# scalar regularization parameter would be. Note that in practice, you would obviously not be able
# to obtain this reconstruction since the target image is not available. However, the comparison
# is useful to assess how much improvement one can expect to obtain when employing
# spatially adaptive regularization parameter maps.


# %%
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


regularization_parameters = torch.linspace(0.0002, 0.00075, 10)
mse_values, pdhg_recon_best_scalar = line_search(
    regularization_parameters,
    pseudo_inverse_solution_undersampled,
    kdata_undersampled,
    forward_operator_undersampled,
    (image_fully_sampled.data).cuda() if torch.cuda.is_available() else image_fully_sampled.data,
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
# Finally, let us compare the different reconstructions. We show the adjoint reconstruction, the pseudo-inverse, the
# PDHG reconstruction with the best scalar regularization parameter, the PDHG reconstruction with the spatially adaptive
# regularization parameter map and the iterative SENSE reconstruction from the fully-sampled data.


# %%
show_images(
    adjoint_recon_undersampled.squeeze(),
    pseudo_inverse_solution_undersampled.squeeze(),
    pdhg_recon_best_scalar.squeeze(),
    pdhg_recon_regularization_parameter_trained_.squeeze(),
    image_fully_sampled.data.squeeze(),
    titles=[
        'Adjoint',
        'Pseudo-Inverse',
        r'PDHG (Best $\lambda>0$)',
        r'PDHG (Trained $\Lambda_{\theta}$-Map)',
        'Iterative SENSE \n (Fully-Sampled)',
    ],
    show_mse=True,
    clim=(0.0, 0.4 * image_fully_sampled.data.abs().max().item()),
)

# %%

show_images(
    regularization_parameter_map_undersampled_data[0, 0].squeeze(),
    titles=[
        r'$\Lambda_{\theta}$-Parameter Map',
    ],
    cmap='inferno',
    clim=(0.0, regularization_parameter_map_trained.abs().max().item()),
    colorbar=True,
)

# %% [markdown]
# Well done, we have successfully reconstructed an image with spatially varying regularization parameter
# maps for TV. 🎉


# ## Next steps
# As previously mentioned, you can also change network architecture to something more sophisticated.
# Do deeper/wider networks give more accurate results?
# Further, you can play around with the number of iterations used for unrolling PDHG at training time. How does this
# number of iterations influence the obtained lambda maps and the final reconstruction?
# Further, since PDHG is a convergent method, you can also let the number of iterations of PDHG go to
# infinity at test time.

# %%
