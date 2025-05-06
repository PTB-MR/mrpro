# %% [markdown]
# # Learning spatially adaptive regularization parameter maps for total-variation (TV)-minimization reconstruction

# %% [markdown]
# ## Overview
# In this notebook, we demonstrate how the method presented in
# [[Kofler et al, SIIMS 2023](https://epubs.siam.org/doi/abs/10.1137/23M1552486)] can be implemented for
# 2D MR reconstruction problems using MRpro. The method consists of two main blocks.
# 1) The first block is a neural network architecture that estimates spatially adaptive regularization
# parameter maps, which are then used in 2).
# 2) The second block corresponds to an unrolled unrolled Primal-Dual Hybrid Gradient (PDHG) algorithm
# [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)] that
# reconstruct the image from the undersampled k-space data measurements assuming the regularization
# parameter maps to be fixed.
#
# The entire network can be trained end-to-end using the MRpro framework, allowing to learn to
# estimate spatially adaptive regularization parameter maps from an input image.

# %% [markdown]
# ### The method
# In the TV-example, (see <project:tv_minimization_reconstruction_pdhg.ipynb>), you can see how to employ the primal
# dual hybrid gradient (PDHG) method [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)]
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
# and then to consider the problem
#
# $\mathcal{F}_{\Lambda_{\theta}}(x) = \frac{1}{2}||Ax - y||_2^2 +  \| \Lambda_{\theta} \nabla x \|_1, \quad \quad (2)$
#
# where $\Lambda_{\theta}$ is voxel-wise strictly positive and locally regularizes the TV-minimization problem.


# %% [markdown]
# ### The neural network
# In this simple example, we use a simple convolutional neural network to estimate the regularization parameter maps.
# Obviously, more complex architectures can be employed as well. For example, in the work in
# [[Kofler et al, SIIMS 2023](https://epubs.siam.org/doi/abs/10.1137/23M1552486)],
# a U-Net [[Ronneberger et al, MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)] was used.
# The network used here corresponds to a simple block of convolutional layers with leaky ReLU activations and a
# final softplus activation to ensure that the regularization parameter maps are strictly positive.
# The network is defined in the following.

import torch


class ParameterMapNetwork2D(torch.nn.Module):
    r"""A simple network for estimating regularization parameter maps for TV-reconstruction."""

    def __init__(self, n_filters: int = 16) -> None:
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        n_filters
            number of filters to be applied in the convolutional layers of the network.

        """
        super().__init__()

        self.cnn_block = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=3, stride=1, padding=1),
            ]
        )
        # raw parameter t, softplus is used to "activate" it and make it positive
        self.t = torch.nn.Parameter(torch.tensor([-5.0], requires_grad=True))

        self.beta_softplus = 1.0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""Apply the network to estimate regularization parameter maps.

        Parameters
        ----------
        image
            the image from which the regularization parameter maps should be estimated.

        """
        dims = tuple(dim for dim in range(1, image.ndim))
        image = (image - image.mean(dim=dims, keepdim=True)) / image.std(dim=dims, keepdim=True)
        regularization_parameter_map = self.cnn_block(image)

        # stack the parameter map channel dimension to share the regularization
        # between the x- and y-direction of the image gradients
        regularization_parameter_map = torch.concat(2 * [regularization_parameter_map], dim=1)

        # apply softplus to enforce strict positvity and scale by hand-crafted parameter
        regularization_parameter_map = torch.nn.functional.softplus(
            self.t, beta=self.beta_softplus
        ) * torch.nn.functional.softplus(regularization_parameter_map, beta=self.beta_softplus)
        return regularization_parameter_map


# %% [markdown]
# ### The unrolled PDHG algorithm network
# We now construct the second block, i.e. network that solves the TV-minimization problem with spatially
# adaptive regularization parameter maps given in (2) by unrolling a finite number of iteration of PDHG.

# %% [markdown]
# ```{note}
# To fully understand the mechanics of the network, we recommend to have a look at the TV-example in
# <project:tv_minimization_reconstruction_pdhg.ipynb>.
# ```

# %% [markdown]
# Put in simple words, the network takes the initial image, estimates the regularization
# parameter maps, and then sets up the TV-problem presented in (2) within the "forward" of the network.
# The network then approximately solves the TV-problem using the PDHG algorithm
# and returns the reconstructed image.

# %%
import mrpro


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
        img_shape: mrpro.data.SpatialDimension[int],
        k_shape: mrpro.data.SpatialDimension[int],
        lambda_map_network: torch.nn.Module,
        n_iterations: int = 128,
    ):
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        img_shape
            image shape, i.e. the shape of the domain of the fourier operator.
        k_shape
           k-space shape, i.e. the shape of the domain of the fourier operator
        lambda_map_network
            a network that predicts a regularization parameter map from the input image.
        n_iterations
            number of iterations for the unrolled primal dual hybrid gradient (PDHG) algorithm.

        """
        super().__init__()

        self.img_shape = img_shape
        self.k_shape = k_shape

        self.lambda_map_network = lambda_map_network
        self.n_iterations = n_iterations

        finite_differences_operator = mrpro.operators.FiniteDifferenceOp(dim=(-2, -1), mode='forward')
        gradient_operator = (
            mrpro.operators.RearrangeOp('grad batch ... -> batch grad ... ') @ finite_differences_operator
        )
        self.gradient_operator = gradient_operator

        self.g = mrpro.operators.functionals.ZeroFunctional()

        # operator norm of the stacked operator K=[A, \nabla]^T
        self.stacked_operator_norm = 3.0  # analytically calculated

    def estimate_lambda_map(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate regularization parameter map from image."""
        # squeeze the dimensions that are one. In particular, the initial image
        # is a coil-combined image (i.e. coils=1) and because, we only consider 2D problems here,
        # z can be assumed to be always 1.
        # (other*, coils, z, y, x) -> (other*, y, x)
        input_image = image.abs().squeeze(-3)
        regularization_parameter_map = self.lambda_map_network(input_image).unsqueeze(-3).unsqueeze(-3)
        return regularization_parameter_map

    def forward(
        self,
        initial_image: torch.Tensor,
        csm: torch.Tensor,
        kdata: torch.Tensor,
        traj: mrpro.data.KTrajectory,
        regularization_parameter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct image using an unrolled PDHG algorithm.

        Parameters
        ----------
        initial_image
            initial guess of the solution of the TV problem.
        csm
            coil sensitivity map tensor of the considered problem.
        kdata
            k-space data tensor of the considered problem.
        traj
            k-space trajectory tensor of the Fourier considered operator.
        regularization_parameter
            regularization parameter to be used in the TV-functional. If set to None,
            it is estimated by the lambda_map_network.
            (can also be a single scalar)

        Returns
        -------
            Image reconstructed by the TV-minimization algorithm.
        """
        # if no regularization parameter map is provided, compute it with the network
        if regularization_parameter is None:
            regularization_parameter = self.estimate_lambda_map(initial_image)

        f_1 = 0.5 * mrpro.operators.functionals.L2NormSquared(target=kdata, divide_by_n=False)
        f_2 = mrpro.operators.functionals.L1NormViewAsReal(weight=regularization_parameter, divide_by_n=False)
        f = mrpro.operators.ProximableFunctionalSeparableSum(f_1, f_2)

        fourier_operator = mrpro.operators.FourierOp(
            recon_matrix=self.img_shape,
            encoding_matrix=self.k_shape,
            traj=traj,
        )
        csm_operator = mrpro.operators.SensitivityOp(csm)
        forward_operator = fourier_operator @ csm_operator
        stacked_operator = mrpro.operators.LinearOperatorMatrix(((forward_operator,), (self.gradient_operator,)))

        primal_stepsize = dual_stepsize = 0.95 * 1.0 / self.stacked_operator_norm

        (solution,) = mrpro.algorithms.optimizers.pdhg(
            f=f,
            g=self.g,
            operator=stacked_operator,
            initial_values=(initial_image,),
            max_iterations=self.n_iterations,
            primal_stepsize=primal_stepsize,
            dual_stepsize=dual_stepsize,
            tolerance=0.0,
        )
        return solution


# %% [markdown]
# ### Creating the training data
# In the following, we create some training data for the network. We use some images borrowed from the
# BrainWeb dataset [[Aubert-Broche et al, IEEE TMI 2006](https://ieeexplore.ieee.org/abstract/document/1717639),
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
    kdata: torch.Tensor, target_image: torch.Tensor | None, factor: float = 100
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Normalize k-space data and (possibly) target image.

    Divide by std multiply by (an empirically set) factor.
    """
    kdata_norm = torch.linalg.norm(kdata)

    kdata /= kdata_norm
    kdata *= factor
    if target_image is not None:
        target_image /= kdata_norm
        target_image *= factor

        return kdata, target_image
    else:
        return kdata


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

    def prepare_data(
        self, image: torch.Tensor, n_coils: int, acceleration_factor: int, noise_variance: float, seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, mrpro.data.KTrajectory, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training by simulating k-space data.

        Parameters
        ----------
        image : torch.Tensor
            Input image to be used for simulation.
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
            A tuple containing k-space data, coil sensitivity maps, k-space trajectories adjoint reconstruction,
            pseudo-inverse solution and the image.
        """
        # randomly choose trajectories to define the fourier operator
        ny, nx = image.shape[-2:]
        img_shape = mrpro.data.SpatialDimension(z=1, y=ny, x=nx)
        k_shape = mrpro.data.SpatialDimension(z=1, y=ny, x=nx)

        traj = mrpro.data.traj_calculators.KTrajectoryCartesian().gaussian_variable_density(
            encoding_matrix=k_shape, acceleration=acceleration_factor, n_center=10, fwhm_ratio=0.6, seed=seed
        )

        fourier_operator = mrpro.operators.FourierOp(
            traj=traj,
            recon_matrix=img_shape,
            encoding_matrix=k_shape,
        )

        # generate simualated coil sensitivity maps
        from mrpro.phantoms.coils import birdcage_2d

        with torch.no_grad():
            csm = birdcage_2d(n_coils, img_shape, relative_radius=0.8)
            csm_operator = mrpro.operators.SensitivityOp(csm)

            forward_operator = fourier_operator @ csm_operator

            (kdata,) = forward_operator(image)
            kdata = add_gaussian_noise(kdata, noise_variance=noise_variance, seed=seed)

            assert image is not None
            kdata, image = normalize_kspace_data_and_image(kdata, image)

            (adjoint_recon,) = forward_operator.H(kdata)

            # compute an approximation of the pseudo-inverse solution
            # Note that this in general only makes sense if the forward operator can be injective,
            # i.e. we have multiple coils with acceleration factor < n_coils
            (pseudo_inverse_solution,) = mrpro.algorithms.optimizers.cg(
                forward_operator.gram, right_hand_side=adjoint_recon, initial_value=adjoint_recon, max_iterations=16
            )

        return kdata, csm, traj, adjoint_recon, pseudo_inverse_solution, image

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """Return the image indexed by idx."""
        image = self.images[idx]

        return image


# %% [markdown]
# Download the data from Zenodo and create the datasets and dataloaders.

# %%
import tempfile
from pathlib import Path

import zenodo_get

dataset = '15348116'

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

plt.ioff()
import torch


def show_images(
    *images: torch.Tensor,
    titles: list[str] | None = None,
    cmap: str = 'grey',
    clim: tuple[float, float] | None = None,
    rotate: bool = True,
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
        image_reference = image_reference.rot90(k=-1, dims=(-2, -1))
    for i in range(n_images):
        image = images[i].cpu() if images[i].is_cuda else images[i]
        if rotate:
            image = image.rot90(k=-1, dims=(-2, -1))
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

image_ = next(iter(dataloader_validation))[0]

n_coils = 8
noise_variance = 0.05
acceleration_factor = 4
kdata_, csm_, traj_, adjoint_recon_, pseudo_inverse_solution_, image_ = dataset_validation.prepare_data(
    image_, n_coils=n_coils, acceleration_factor=acceleration_factor, noise_variance=noise_variance
)

show_images(
    adjoint_recon_.squeeze(),
    pseudo_inverse_solution_.squeeze(),
    image_.squeeze(),
    titles=['Adjoint', 'Pseudo-Inverse', 'Target'],
    rotate=False,
    show_mse=True,
    clim=(0, 0.5),
)

# %%[markdown]
# We now define the unrolled PDHG network.

# %%
torch.manual_seed(2025)
lambda_map_network = ParameterMapNetwork2D(n_filters=32)
n_iterations = 128

img_shape = mrpro.data.SpatialDimension(z=1, y=images.shape[-2], x=images.shape[-1])
k_shape = mrpro.data.SpatialDimension(z=1, y=images.shape[-2], x=images.shape[-1])

adaptive_tv_network = AdaptiveTVNetwork2D(
    img_shape=img_shape,
    k_shape=k_shape,
    lambda_map_network=lambda_map_network,
    n_iterations=n_iterations,
)

if torch.cuda.is_available():
    adaptive_tv_network = adaptive_tv_network.cuda()
    pseudo_inverse_solution_ = pseudo_inverse_solution_.cuda()
    csm_ = csm_.cuda()
    kdata_ = kdata_.cuda()
    traj_ = traj_.cuda()

with torch.no_grad():
    regularization_parameter_map_init = adaptive_tv_network.estimate_lambda_map(pseudo_inverse_solution_)

# %% [markdown]
# ## Network training
# Let us now train the unrolled PDHG network. We set up the optimizer with an appropriate learning rate, choose a
# number of epochs and write a simple training loop, in which we present the network input-target pairs and update
# the network's parameters by gradient-descent.
# To obtain a somewhat decent estimate of the network parameters, training should take approximately 5-6 minutes
# on a GPU. If you have GPU and you need a break, start the training now, grab yourself a coffee and come back later
# to see your network trained from scratch.
# If you do not have a GPU, we provide a pre-trained model, which you can load and directly apply to the data.
from typing import cast

cnn_block = cast(torch.nn.Module, adaptive_tv_network.lambda_map_network.cnn_block)
optimizer = torch.optim.Adam(
    [
        {'params': cnn_block.parameters(), 'lr': 1e-4},
        {'params': [adaptive_tv_network.lambda_map_network.t], 'lr': 1e-1},
    ],
    weight_decay=1e-5,
)

import copy
import datetime
from time import time

from tqdm import tqdm

n_epochs = 10
validation_loss_values = []
best_model = None
noise_variance = 0.05
acceleration_factor = 4

time_start = time()

if torch.cuda.is_available():
    for epoch in tqdm(range(n_epochs), desc='epochs', disable=False):
        for sample_num, image_ in enumerate(dataloader_train):
            optimizer.zero_grad()

            kdata, csm, traj, adjoint_recon, pseudo_inverse_solution, image = dataset_train.prepare_data(
                image_,
                n_coils=n_coils,
                acceleration_factor=acceleration_factor,
                noise_variance=noise_variance,
                seed=sample_num * epoch,
            )

            if torch.cuda.is_available():
                kdata = kdata.cuda()
                traj = traj.cuda()
                csm = csm.cuda()
                pseudo_inverse_solution = pseudo_inverse_solution.cuda()
                image = image.cuda()

            pdhg_recon = adaptive_tv_network(pseudo_inverse_solution, csm, kdata, traj)

            loss = torch.nn.functional.mse_loss(torch.view_as_real(pdhg_recon), torch.view_as_real(image))

            loss.backward()
            optimizer.step()

        seed_validation = 0
        running_loss = 0.0
        for sample_num, image_ in enumerate(dataloader_validation):
            kdata, csm, traj, adjoint_recon, pseudo_inverse_solution, image = dataset_validation.prepare_data(
                image_,
                n_coils=n_coils,
                acceleration_factor=acceleration_factor,
                noise_variance=noise_variance,
                seed=sample_num,
            )

            if torch.cuda.is_available():
                kdata = kdata.cuda()
                traj = traj.cuda()
                csm = csm.cuda()
                pseudo_inverse_solution = pseudo_inverse_solution.cuda()
                image = image.cuda()

            with torch.no_grad():
                pdhg_recon = adaptive_tv_network(pseudo_inverse_solution, csm, kdata, traj)
                loss = torch.nn.functional.mse_loss(torch.view_as_real(pdhg_recon), torch.view_as_real(image))

            running_loss += loss.item()
        loss_validation = running_loss / len(dataset_validation)
        validation_loss_values.append(loss_validation)
        print(validation_loss_values[-1])
        if validation_loss_values[-1] <= min(validation_loss_values):
            # store the weights if the validation loss is the smallest; this is
            # to avoid possible overfitting here, since we are only using very few
            # images
            best_model = copy.deepcopy(adaptive_tv_network.state_dict())

    assert best_model is not None
    adaptive_tv_network.load_state_dict(best_model)

    fig, ax = plt.subplots()
    ax.plot(range(1, n_epochs + 1), validation_loss_values, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()

    time_end = time() - time_start
    print(f'training time: {datetime.timedelta(seconds=time_end)}')
else:
    adaptive_tv_network.load_state_dict(torch.load(data_folder_tv / 'tv_model.pt', map_location='cpu'))

# %% [markdown]
# Nice! We have now trained our network for estimating the regularization parameter maps. Let us check if the obtained
# maps show interesting features by applying it to the previous image.
with torch.no_grad():
    regularization_parameter_map_trained = adaptive_tv_network.estimate_lambda_map(pseudo_inverse_solution_)

    pdhg_recon_regularization_parameter_trained_ = adaptive_tv_network(
        pseudo_inverse_solution_, csm_, kdata_, traj_, regularization_parameter_map_trained
    )

# %%
show_images(
    regularization_parameter_map_init[0, 0].squeeze(),
    regularization_parameter_map_trained[0, 0].squeeze().cpu(),
    titles=[
        'Reg. Parameter Map \n (Before Training)',
        'Reg. Parameter Map \n (After Training)',
    ],
    cmap='inferno',
    clim=(0.0, regularization_parameter_map_trained.abs().max().item()),
    rotate=False,
    colorbar=True,
)


# %% [markdown]
## Application to in-vivo data
# Let us now download some real measured scanner data and apply our trained network. We will use the same dataset of the
# Cartesian reconstruction example <project:cartesian_reconstruction.ipynb>. From the measured data, we select
# a subset of the sampled k-space lines to create an undersampled k-space data and reconstruct it with the just
# trained method.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Get the raw data from zenodo
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

# %%
nz, ny, nx = kdata_cartesian.data.shape[-3:]
k_shape_cartesian = mrpro.data.SpatialDimension(z=nz, y=ny, x=nx)
n_k1 = kdata_cartesian.data.shape[-2]
k1_center = n_k1 // 2

ktraj_undersampled = mrpro.data.traj_calculators.KTrajectoryCartesian().gaussian_variable_density(
    encoding_matrix=k_shape_cartesian, acceleration=acceleration_factor, n_center=10, fwhm_ratio=0.6, seed=2024
)

k1_idx = ktraj_undersampled.ky.squeeze() + k1_center
kdata_undersampled = kdata_cartesian.data[..., k1_idx.to(torch.int), :]

kdata_undersampled_ = normalize_kspace_data_and_image(kdata_undersampled, None)
assert isinstance(kdata_undersampled_, torch.Tensor)
kdata_undersampled = kdata_undersampled_

fourier_operator_undersampled = mrpro.operators.FourierOp(
    traj=ktraj_undersampled,
    recon_matrix=kdata_cartesian.header.recon_matrix,
    encoding_matrix=kdata_cartesian.header.encoding_matrix,
)
(coil_images,) = fourier_operator_undersampled.H(kdata_undersampled)
# %%
# estimate coil sensitivity maps using the Walsh method in MRpro
csm_undersampled = mrpro.algorithms.csm.walsh(coil_images.squeeze(0), smoothing_width=5).unsqueeze(0)


# %%
forward_operator = fourier_operator_undersampled @ mrpro.operators.SensitivityOp(csm_undersampled)
(adjoint_recon_undersampled,) = forward_operator.H(kdata_undersampled)

(pseudo_inverse_solution_undersampled,) = mrpro.algorithms.optimizers.cg(
    forward_operator.gram,
    right_hand_side=adjoint_recon_undersampled,
    initial_value=adjoint_recon_undersampled,
    max_iterations=16,
)

if torch.cuda.is_available():
    pseudo_inverse_solution_undersampled = pseudo_inverse_solution_undersampled.cuda()
    adjoint_recon_undersampled = adjoint_recon_undersampled.cuda()
    csm_undersampled = csm_undersampled.cuda()
    ktraj_undersampled = ktraj_undersampled.cuda()
    kdata_undersampled = kdata_undersampled.cuda()

# %%

adaptive_tv_network.img_shape = kdata_cartesian.header.recon_matrix
adaptive_tv_network.k_shape = kdata_cartesian.header.encoding_matrix

with torch.no_grad():
    regularization_parameter_map_undersampled_data = adaptive_tv_network.estimate_lambda_map(
        pseudo_inverse_solution_undersampled
    )

    pdhg_recon_regularization_parameter_trained_ = adaptive_tv_network(
        pseudo_inverse_solution_undersampled,
        csm_undersampled,
        kdata_undersampled,
        ktraj_undersampled,
        regularization_parameter_map_undersampled_data,
    )

# %% [markdown]
# Finally, let us have a look at the reconstruced image as well as as the estimate regularization parameter
# map. Also, let us compare the reconstructed image to the one reconstructed from the fully-sampled k-space data using
# the iterative SENSE reconstruction class.

# %%
# Create DirectReconstruction object from KData object. Also, scale the k-space data to better match the
# intensities of the undersampled k-space data
kdata_cartesian_data_normalized = normalize_kspace_data_and_image(kdata_cartesian.data, None)
assert isinstance(kdata_cartesian_data_normalized, torch.Tensor)
kdata_cartesian.data = kdata_cartesian_data_normalized
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_cartesian)
image_fully_sampled = iterative_sense_reconstruction(kdata_cartesian)


# %% [markdown]
# We also perform a quick line search to see what the best possible TV-reconstruction using a
# scalar regularization parameter would be. Note that in practice, you would obviously not be able
# to obtain this reconstruction since the target image is not available. However, the comparison
# is useful to assess how much improvement one can expect to obtain when employing
# spatially adaptive regularization parameter maps.
def line_search(
    regularization_parameters: torch.Tensor,
    initial_image: torch.Tensor,
    csm: torch.Tensor,
    kdata: torch.Tensor,
    traj: mrpro.data.KTrajectory,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform a line search to pick the best regularization parameter for TV."""
    reconstructions_list = [
        adaptive_tv_network(initial_image, csm, kdata, traj, regularization_parameter)
        for regularization_parameter in regularization_parameters
    ]
    mse_values = torch.tensor(
        [
            torch.nn.functional.mse_loss(torch.view_as_real(recon), torch.view_as_real(target))
            for recon in reconstructions_list
        ]
    )

    return mse_values, reconstructions_list[torch.argmin(torch.tensor(mse_values))]


regularization_parameters = torch.linspace(0.014, 0.022, 10)
mse_values, pdhg_recon_best_scalar = line_search(
    regularization_parameters,
    pseudo_inverse_solution_undersampled,
    csm_undersampled,
    kdata_undersampled,
    ktraj_undersampled,
    (image_fully_sampled.data).cuda() if torch.cuda.is_available() else image_fully_sampled.data,
)

fig, ax = plt.subplots()
ax.plot(regularization_parameters, mse_values.cpu())
ax.set_xlabel(r'Scalar Regularization Parameter Value $\lambda$', fontsize=12)
ax.set_ylabel(r'MSE(TV($\lambda$),Target)', fontsize=12)
ax.vlines(
    x=regularization_parameters[torch.argmin(mse_values)].item(),
    ymin=mse_values.min().item(),
    ymax=mse_values.max().item(),
    colors='red',
    ls=':',
    label=r'Best scalar $\lambda>0$',
)
ax.legend()
plt.show()

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
    rotate=False,
    show_mse=True,
    clim=(0.0, 1.0),
)

# %%
show_images(
    regularization_parameter_map_undersampled_data[0, 0].squeeze(),
    titles=[
        r'$\Lambda_{\theta}$-Parameter Map',
    ],
    cmap='inferno',
    clim=(0.0, regularization_parameter_map_trained.abs().max().item()),
    rotate=False,
    colorbar=True,
)

# %% [markdown]
# Well done, we have successfully reconstructed an image with spatially varying regularization parameter
# maps for TV.
#
# ### Next steps
# As previously mentioned, you can also change network architecture to something more sophisticated.
# Do deeper/wider networks give more accurate results?
# Further, you can play around with the number of iterations used for unrolling PDHG at training time. How does this
# number of iterations influence the obtained lambda maps and the final reconstruction?
# Further, since PDHG is a convergent method, you can also let the number of iterations of PDHG go to
# infinity, at test time.

# %%
