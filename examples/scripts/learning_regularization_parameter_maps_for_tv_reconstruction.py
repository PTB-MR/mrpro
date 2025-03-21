# %% [markdown]
# # Learning spatially adaptive regularization parameter maps for total-variation (TV)-minimization reconstruction

# %% [markdown]
# ## Overview
# In this notebook, we are going to demonstrate how the method presented in
# [[Kofler et al, SIIMS 2023](https://epubs.siam.org/doi/abs/10.1137/23M1552486)] can be implemented for
# 2D MR reconstruction problems using MRpro. The method consists of two main blocks.
# 1) The first block is a neural network architecture that estimates spatially adaptive regularization
# parameter maps, which are then used in 2).
# 2) The second block corresponds to an unrolled unrolled Primal-Dual Hybrid Gradient (PDHG) algorithm to
# reconstruct an images from undersampled k-space data # measurements assuming the regularization
# parameter maps are to be fixed.
# The entire pipeline can trained end-to-end using the MRpro framework, allowing to learn to
# estimate spatially adaptive regularization parameter maps.

# %% [markdown]
# ### The method
# In the TV-example, (see <project:tv_minimization_reconstruction_pdhg.ipynb>), we have seen how to employ the primal
# dual hybrid gradient (PDHG) method # [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)]
# to solve the TV-minimization problem. There, for data acquired according to the usual model
#
# $ y = Ax_{\mathrm{true}} + n, $
#
# where $A$ contains the Fourier transform and the coil sensitivity maps operator, etc, and $n$ is complex-valued
# Gaussian noise, the TV-minimization problem was given as
#
# $\mathcal{F}_{\lambda}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| \nabla x \|_1, \quad \quad \quad (1)$
#
# where $\nabla$ is the discretized gradient operator and $\lambda>0$ globally dictates the strength of the
# regularization. Clearly, having one global regularization parameter $\lambda$ for the entire image is
# not optimal, as the image content can vary significantly across the image. Therefore, in this example,
# we aim to learn spatially adaptive regularization parameter maps $\Lambda$ from the input image to improve
# our TV-reconstruction. I.e., we are interested in estimating spatially adaptive regularization parameter maps by
#
# $\Lambda_{\theta}:=u_{\theta}(x_0)$
#
# with a convolutional neural network $u_{\theta}$ with trainable parameters $\theta$, and then to consider the problem
#
# $\mathcal{F}_{\Lambda_{\theta}}(x) = \frac{1}{2}||Ax - y||_2^2 +  \| \Lambda_{\theta} \nabla x \|_1, \quad \quad (2)$
#
# where $\Lambda$ is point-wise strictly positive and locally regularizes the TV-minimization problem.


# %% [markdown]
# ### The neural network
# In this simple example, we use a simple convolutional neural network to estimate the regularization parameter maps.
# Obviously, more complex architectures can be employed as well. For example, in the work in
# [[Kofler et al, SIIMS 2023](https://epubs.siam.org/doi/abs/10.1137/23M1552486)],
# a U-Net [[Ronneberger et al, MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)] was used.
# The network used here corresponds to a simple block of convolutional layers with leaky ReLU activations and a
# final softplus activation to ensure # that the regularization parameter maps are strictly positive.
# The network is defined in the following.

import torch
import torch.nn as nn
from einops import rearrange


class ParameterMapNetwork2D(nn.Module):
    r"""A simple network for estimating regularization parameter maps for TV-reconstruction."""

    def __init__(self, n_filters: int = 16) -> None:
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        n_filters
            number of filters to be applied in the convolutional layers of the network.

        """
        super().__init__()

        self.cnn_block = nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.t = 0.1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""Apply the network to estimate regularization parameter maps.

        Parameters
        ----------
        image
            the image from which the regularization parameter maps should be estimated.

        """
        regularization_parameter_map = self.cnn_block(image)

        # stack the parameter map channel dimension to share the regularization
        # between the x- and y-direction of the image gradients
        regularization_parameter_map = torch.concat(2 * [regularization_parameter_map], dim=1)

        # apply softplus to enforce strict positvity and scale by hand-crafted parameter
        regularization_parameter_map = self.t * torch.nn.functional.softplus(regularization_parameter_map)
        return regularization_parameter_map


# %% [markdown]
# ### The unrolled PDHG algorithm network
# We now construct the network that solves the TV-minimization problem with spatially adaptive regularization
# parameter maps by unrolling a finite number of iteration of PDHG.
# The network is defined in the following.

# %% [markdown]
# ```{note}
# To fully understand the mechanics of the network, we recommend to have a look at the TV-example in
# <project:tv_minimization_reconstruction_pdhg.ipynb>.
# ```

# %% [markdown]
# However, put in simple words, the network takes the initial image, estimates the regularization
# parameter maps, and then sets up the # TV-problem within the "forward" of the network. The network
# then approximately solves the TV-problem using the PDHG algorithm # and returns the reconstructed image.

import mrpro
from mrpro.operators.LinearOperator import LinearOperator


class AdaptiveTVNetwork2D(nn.Module):
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
        fourier_operator: LinearOperator,
        gradient_operator: LinearOperator,
        lambda_map_network: torch.nn.Module,
        n_iterations: int = 128,
    ):
        r"""Initialize Adaptive TV Network.

        Parameters
        ----------
        fourier_operator
            fourier_operator of the problem :math: `y=Ax + e` with :math:`A:=FS`,
            where :math:`F` is the Fourier operator and :math:`S` is the sensitivity operator.
        gradient_operator
           the gradient operator to be used in the :math: `\ell_1`-norm of the TV-problem.
        lambda_map_network
            a network that predicts a regularization parameter map from the input image.
        n_iterations
            number of iterations for the unrolled primal dual hybrid gradient (PDHG) algorithm.

        """
        super().__init__()
        self.fourier_operator = fourier_operator
        self.gradient_operator = gradient_operator
        self.lambda_map_network = lambda_map_network
        self.n_iterations = n_iterations
        self.g = mrpro.operators.functionals.ZeroFunctional()

        # the operator norm of the stacked operator K = [fourier, gradient]^T;
        # can be explicitly calculated
        # self.register_buffer('operator_norm_K', torch.tensor([3.0]))
        self.operator_norm_K = 3.0

    def estimate_lambda_map(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate regularization parameter map from image."""
        # squeeze the dimensions that are one. In particular, the initial image
        # is a coil-combined image (i.e. coils=1) and because, we only consider 2D problems here,
        # z can be assumed to be always 1.
        # (other*, coils, z, y, x) -> (other*, y, x)
        input_image = image.squeeze(-3).squeeze(-3)
        input_image = rearrange(torch.view_as_real(input_image), 'other y x ch -> other ch y x')
        regularization_parameter_map = self.lambda_map_network(input_image).unsqueeze(-3).unsqueeze(-3)
        return regularization_parameter_map

    def forward(
        self,
        initial_image: torch.Tensor,
        csm: torch.Tensor,
        kdata: torch.Tensor,
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

        # set up the problem K = [A, \nabla]^T, f=f1+f2, g=0
        csm_operator = mrpro.operators.SensitivityOp(csm).to(initial_image.device)
        forward_model = self.fourier_operator @ csm_operator
        self.K_op = mrpro.operators.LinearOperatorMatrix(((forward_model,), (self.gradient_operator,)))
        f_1 = 0.5 * mrpro.operators.functionals.L2NormSquared(target=kdata, divide_by_n=False)
        f_2 = mrpro.operators.functionals.L1NormViewAsReal(weight=regularization_parameter, divide_by_n=False)
        f = mrpro.operators.ProximableFunctionalSeparableSum(f_1, f_2)

        primal_stepsize = dual_stepsize = 0.95 * 1.0 / self.operator_norm_K

        (solution,) = mrpro.algorithms.optimizers.pdhg(
            f=f,
            g=self.g,
            operator=self.K_op,
            initial_values=(initial_image,),
            max_iterations=self.n_iterations,
            primal_stepsize=primal_stepsize,
            dual_stepsize=dual_stepsize,
            tolerance=0.0,
        )
        return solution


# %%
## %% [markdown]
# ### Creating training data
# In the following, we create some training data for the network. We use some images borrowed from the
# BrainWeb dataset [[Aubert-Broche et al, IEEE TMI 2006](https://ieeexplore.ieee.org/abstract/document/1717639),
# which is a simulated MRI dataset and for which MRpro provides a simple interface to load the data, see ...
# For simplicity, we here use a set of XXX images that were generated by simulating the contrast using the
# inversion recovery signl model in ...
# First, we start by defining some auxiliary functions that we will need for retrospectively simulating undersampled
# and noiy k-space data.
def normal_pdf(length: int, sensitivity: float) -> torch.Tensor:
    """Return the normal probability density function."""
    return torch.exp(-sensitivity * (torch.arange(length) - length / 2) ** 2)


def choose_kspace_lines(n_samples: int, acceleration_factor: float, sample_n: int = 10) -> torch.Tensor:
    """Choose k-space lines according to Gaussian."""
    n_to_keep = int(n_samples // acceleration_factor)
    pdf_x = normal_pdf(n_samples, 0.5 / (n_samples / 10.0) ** 2)
    lmda = n_samples / (2.0 * acceleration_factor)
    # add uniform distribution
    pdf_x += lmda * 1.0 / n_samples
    if sample_n:
        pdf_x[n_samples // 2 - sample_n // 2 : n_samples // 2 + sample_n // 2] = 0
        pdf_x /= torch.sum(pdf_x)
        n_to_keep -= sample_n
    indices = pdf_x.multinomial(num_samples=n_samples, replacement=False)
    k1_idx = torch.arange(0, n_samples)[indices[:n_to_keep]]
    if sample_n:
        center_indices = torch.arange(n_samples // 2 - sample_n // 2, n_samples // 2 + sample_n // 2)
        k1_idx = torch.cat([k1_idx, center_indices], dim=-1)
    return torch.sort(k1_idx)[0]


from collections.abc import Sequence


def add_gaussian_noise(kdata: torch.Tensor, dim: Sequence[int] | None, noise_variance: float) -> torch.Tensor:
    """Corrupt given k-space data with additive Gaussian noise."""
    dim = tuple(dim) if dim is not None else dim
    mu, std = kdata.mean(dim=dim, keepdim=True), kdata.std(dim=dim, keepdim=True)
    kdata = (kdata - mu) / std
    kdata = kdata + noise_variance * torch.randn_like(kdata)
    kdata = kdata * std + mu
    return kdata


class BrainWeb2D(torch.utils.data.Dataset):
    r"""BrainWeb dataset for 2D problems."""

    def __init__(self, images: torch.Tensor) -> None:
        """Parameters

        ----------
        images
            images to be used in the dataset.
        """
        # unsqueeze (N,ny,nx) images to match the MRpro convention (others, coils, z, y, x)
        self.images = images.unsqueeze(-3).unsqueeze(-3)
        self.ny, self.nx = self.images.shape[-2:]
        self.n_k1, self.n_k0 = self.ny, self.nx
        self.im_shape = mrpro.data.SpatialDimension(z=1, y=self.ny, x=self.nx)
        self.k_shape = mrpro.data.SpatialDimension(z=1, y=self.n_k1, x=self.n_k0)

    def prepare_data(
        self, image: torch.Tensor, n_coils: int, acceleration_factor: float, noise_variance: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LinearOperator]:
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

        Returns
        -------
        tuple
            A tuple containing k-space data, coil sensitivity maps, adjoint reconstruction,
            pseudo-inverse solution, original image, and the Fourier operator.
        """
        # randomly choose trajectories to define the fourier operator
        k1_idx = choose_kspace_lines(self.n_k1, acceleration_factor=acceleration_factor)
        traj = mrpro.data.traj_calculators.KTrajectoryCartesian()(
            n_k0=self.n_k0,
            k0_center=self.n_k0 // 2,
            k1_idx=k1_idx[:, None],
            k1_center=self.n_k1 // 2,
            k2_idx=torch.tensor(0),
            k2_center=0,
        )

        fourier_operator = mrpro.operators.FourierOp(
            traj=traj,
            recon_matrix=self.im_shape,
            encoding_matrix=self.k_shape,
        )

        # generate simualated coil sensitivity maps
        from mrpro.phantoms.coils import birdcage_2d

        csm = birdcage_2d(n_coils, self.im_shape, relative_radius=0.8)
        csm_operator = mrpro.operators.SensitivityOp(csm)

        forward_operator = fourier_operator @ csm_operator

        (kdata,) = forward_operator(image)
        kdata = add_gaussian_noise(kdata, dim=(-2, -1), noise_variance=noise_variance)
        (adjoint_recon,) = forward_operator.H(kdata)

        pseudo_inverse_solution = mrpro.algorithms.optimizers.cg(
            forward_operator.gram, right_hand_side=adjoint_recon, initial_value=adjoint_recon, max_iterations=16
        )

        return kdata, csm, adjoint_recon, pseudo_inverse_solution, image, fourier_operator

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """Return the image and retrospectively generate k-space data."""
        image = self.images[idx]

        return image


# %%
images = torch.load('/echo/kofler01/brainweb_data/processed/brainweb_data.pt')
images = torch.view_as_complex(
    rearrange(
        torch.nn.functional.max_pool2d(
            rearrange(torch.view_as_real(images), 'b y x ri -> b ri y x'), kernel_size=2, stride=2
        ),
        'b ri y x -> b y x ri',
        ri=2,
    ),
)
images = torch.rot90(images, k=-1, dims=(-2, -1))

# %%
n_train = 120
images_train = images[:n_train, ...]
images_validation = images[n_train:, ...]

dataset_train = BrainWeb2D(images=images_train)
dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True)

dataset_validation = BrainWeb2D(images=images_validation)
dataloader_validation = torch.utils.data.DataLoader(dataset_train, shuffle=False)

# %% [markdown]
# Let's have a look at an example of the training data.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
import matplotlib.pyplot as plt
import torch


def show_images(*images: torch.Tensor, titles: list[str] | None = None, cmap: str = 'grey') -> None:
    """Plot images."""
    n_images = len(images)
    _, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 3, 3))
    for i in range(n_images):
        axes[0][i].imshow(images[i].rot90(k=-1, dims=(-2, -1)), cmap=cmap)
        axes[0][i].axis('off')
        if titles:
            axes[0][i].set_title(titles[i])
    plt.show()


_, image = next(enumerate(dataloader_validation))

n_coils = 8
noise_variance = 0.05
acceleration_factor = 4
kdata, csm, adjoint_recon, pseudo_inverse_solution, image, fourier_operator = dataset_validation.prepare_data(
    image, n_coils=n_coils, acceleration_factor=acceleration_factor, noise_variance=noise_variance
)

show_images(
    adjoint_recon.abs().squeeze(),
    pseudo_inverse_solution.abs().squeeze(),
    image.abs().squeeze(),
    titles=['adjoint', 'pseudo-inverse', 'target'],
)

# %%[markdown]
# We now construct the unrolled PDHG network. Let us also first have a look at what the current estimates
# of the regularization parameter maps look like prior to training.

# construct gradient operator; also use the Rearrange operator to bring
finite_differences_operator = mrpro.operators.FiniteDifferenceOp(dim=(-2, -1), mode='forward')
gradient_operator = mrpro.operators.RearrangeOp('grad batch ... -> batch grad ... ') @ finite_differences_operator
lambda_map_network = ParameterMapNetwork2D(n_filters=16)
n_iterations = 16
adaptive_tv_network = AdaptiveTVNetwork2D(
    fourier_operator=fourier_operator,
    gradient_operator=gradient_operator,
    lambda_map_network=lambda_map_network,
    n_iterations=n_iterations,
)

# %%
with torch.no_grad():
    regularization_parameter_map = adaptive_tv_network.estimate_lambda_map(pseudo_inverse_solution)

    regularization_parameter_scalar = torch.tensor(0.01)

    pdhg_recon_scalar = adaptive_tv_network(pseudo_inverse_solution, csm, kdata, regularization_parameter_scalar)

show_images(
    adjoint_recon.abs().squeeze(),
    pseudo_inverse_solution.abs().squeeze(),
    pdhg_recon_scalar.abs().squeeze(),
    image.abs().squeeze(),
    titles=['adjoint', 'pseudo-inverse', 'pdhg (scalar reg)', 'target'],
)
show_images(
    regularization_parameter_map[0, 0].abs().squeeze(), titles=['spatial regularization parameter map'], cmap='inferno'
)
# %%
learning_rate = 1e-4
optimizer = torch.optim.Adam(adaptive_tv_network.parameters(), lr=learning_rate)
# loss_function = mrpro.operators.functionals.MSE(image) # todo
loss_function = nn.MSELoss()
n_epochs = 36

loss_vals = []

if torch.cuda.is_available():
    adaptive_tv_network = adaptive_tv_network.cuda()


# %%
for _ in range(n_epochs):
    for _, image in enumerate(dataloader_train):
        print(_)

        optimizer.zero_grad()
        noise_variance = 0.05
        acceleration_factor = 4
        kdata, csm, adjoint_recon, pseudo_inverse_solution, image, fourier_operator = dataset_train.prepare_data(
            image, n_coils=n_coils, acceleration_factor=acceleration_factor, noise_variance=noise_variance
        )

        if torch.cuda.is_available():
            pseudo_inverse_solution = pseudo_inverse_solution.cuda()
            csm = csm.cuda()
            kdata = kdata.cuda()
            image = image.cuda()
            fourier_operator = fourier_operator.cuda()

        adaptive_tv_network.fourier_operator = fourier_operator
        pdhg_recon = adaptive_tv_network(pseudo_inverse_solution, csm, kdata)

        loss = loss_function(torch.view_as_real(pdhg_recon), torch.view_as_real(image))

        loss.backward()
        optimizer.step()
        print(loss.item())
        loss_vals.append(loss.item())

# %% [markdown]
# Well done, we have successfully reconstructed an image with spatially adaptive regularization parameter
# maps for TV.
#
# ### Next steps
# As previously mentioned, you can change network architecture. Do deeper/wider networks give more accurate results?
# Further, you can play around with the number of iterations used for unrolling PDHG at training time. How does this
# number of # iterations the obtained lambda maps and the final reconstruction?
