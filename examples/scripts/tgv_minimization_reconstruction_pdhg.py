# %% [markdown]
# # Total-generalized-variation (TGV)-minimization reconstruction

# %% [markdown]
# In this notebook we demonstrate TGV-based variational reconstruction with a primal-dual hybrid gradient (PDHG) solver
# from **mrpro**.
#
# We work through two examples:
# 1. **Denoising** of a piecewise-constant “square” image — to show how TGV suppresses the staircasing artifact common
# with TV.
# 2. **MRI reconstruction** from retrospectively undersampled 2-D **radial** spokes on a Cartesian grid using
# $A = M \circ \mathcal{F}$.
# %% [markdown]
# ## First Example: Denoising
# %% [markdown]
# ### Load data
# We use the classic square test image (grayscale, scaled to $[0,1]$) and a noisy version with additive white Gaussian
# noise of standard deviation $\sigma \approx 0.05$:
# - `square.png` (clean)
# - `square_noisy_0_05.png` (noisy)
#
# The goal is to recover a clean image from the noisy observation using TV and then TGV for comparison.
# %% mystnb={"code_prompt_show": "Show download details"} tags=["hide-cell"]
# Download raw data from Zenodo
import tempfile
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.set_default_device('cuda')

import zenodo_get

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.download(record='16811276', retry_attempts=5, output_dir=data_folder)

# %% mystnb={"code_prompt_show": "Show plotting details"} tags=["hide-cell"]
import matplotlib.pyplot as plt


def show_images(
    *images: torch.Tensor, titles: list[str] | None = None, clim: tuple[float, float] | None = None
) -> None:
    """Plot images."""
    n_images = len(images)
    _, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 3, 3))
    for i in range(n_images):
        axes[0][i].imshow(images[i].cpu().squeeze(), cmap='gray', clim=clim)
        axes[0][i].axis('off')
        if titles:
            axes[0][i].set_title(titles[i])
    plt.show()


# %% [markdown]
# Display the noisy input and the clean reference for visual comparison.
# %%
import mrpro
from PIL import Image
from torchvision.transforms.functional import to_tensor

square_clean = to_tensor(Image.open(data_folder / 'square.png').convert('L')).to(device=torch.get_default_device())
square_noisy = to_tensor(Image.open(data_folder / 'square_noisy_0_05.png').convert('L')).to(
    device=torch.get_default_device()
)

show_images(square_noisy, square_clean, titles=['Noisy', 'Clean'], clim=(0, 1))

# %% [markdown]
# Before introducing TGV, we first solve a TV-regularized problem as a baseline
# (see [tv_minimization_reconstruction_pdhg.ipynb](examples/notebooks/tv_minimization_reconstruction_pdhg.ipynb)
# for details).
#
# With an observation $y$ and acquisition operator $A$, the TV model reads
# $$
# \min_x\; \tfrac12\,\lVert Ax - y\rVert_2^2\; +\; \lambda\,\lVert \nabla x\rVert_1,
# $$
# where $\nabla$ is the (forward) finite-difference gradient. For denoising we take $A = I$,
# so the first term penalizes the pixel-wise squared error against $y$.
# %% [markdown]
# ### Set up the operator $A$
# For denoising, the acquisition is the identity operator $A = I$ (implemented with `mrpro.operators.IdentityOp`).
# %%
identity_op = mrpro.operators.IdentityOp()  # acquisition operator A here is simply the identity operator

# %% [markdown]
# Here we define a small helper that runs PDHG for TV. The three key components are:
# - Data term: $\tfrac12\lVert Ax - y\rVert_2^2$
# - Gradient term (Regularization term): $\lambda\lVert \nabla x\rVert_1$
# - Operator: $Kx = \begin{bmatrix}Ax\\ \nabla x\end{bmatrix}$, handled with `LinearOperatorMatrix`
#
# We use the adjoint reconstruction as the initial point.
# %%
from collections.abc import Sequence

from mrpro.operators import (
    FiniteDifferenceOp,
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctionalSeparableSum,
    RearrangeOp,
    ZeroOp,
)
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared


def tv_minimization_reconstruction(
    measurement: torch.Tensor,
    acquisition_op: LinearOperator,
    grad_term_weight: torch.Tensor,
    dim: Sequence[int] = (-2, -1),
    **pdhg_kwargs,
) -> torch.Tensor:
    """Perform TV-minimization reconstruction."""
    # 1. Compute initial values using the adjoint of the acquisition operator
    adjoint_recon = acquisition_op.adjoint(measurement)[0]
    initial_values = (adjoint_recon,)

    # 2. Define the objective functional
    data_term = 0.5 * L2NormSquared(target=measurement)
    grad_term = L1NormViewAsReal(weight=grad_term_weight)
    minimization_sum = ProximableFunctionalSeparableSum(data_term, grad_term)

    # 3. Define the operator matrix K
    data_term_row = (acquisition_op,)
    nabla = FiniteDifferenceOp(dim=dim, mode='forward')
    grad_term_row = (nabla,)
    operator_matrix = LinearOperatorMatrix((data_term_row, grad_term_row))

    return mrpro.algorithms.optimizers.pdhg(
        f=minimization_sum, g=None, operator=operator_matrix, initial_values=initial_values, **pdhg_kwargs
    )[0]


# %% [markdown]
# ### Run PDHG with TV

# %% [markdown]
# Let us run PDHG for a number of iterations. The regularization parameter $\lambda = 0.05$ was retrospectively picked
# to achieve a good reconstruction for this example. We can later tune the regularization parameter $\lambda$ and the
# number of iterations to balance smoothing with detail preservation.
# %% mystnb={"code_prompt_show": "Show callback details"} tags=["hide-cell"]
# This is a "callback" function that will be called after each iteration of the PDHG algorithm.
# We use it here to print progress information.

from mrpro.algorithms.optimizers.pdhg import PDHGStatus


def callback(optimizer_status: PDHGStatus) -> None:
    """Print the value of the objective functional every 16th iteration."""
    iteration = optimizer_status['iteration_number']
    solution = optimizer_status['solution']
    if iteration % 16 == 0:
        print(f'Iteration {iteration: >3}: Objective = {optimizer_status["objective"](*solution).item():.3e}')


# %%
square_tv_denoised = tv_minimization_reconstruction(
    measurement=square_noisy,
    acquisition_op=identity_op,
    grad_term_weight=torch.tensor(0.05),
    max_iterations=257,
    callback=callback,
)
show_images(square_noisy, square_tv_denoised, square_clean, titles=['Noisy', 'TV Denoised', 'Clean'], clim=(0, 1))


# %% [markdown]
# TV reduces noise but exhibits staircasing (piecewise-constant plateaus with sharp steps). Next we apply TGV and show
# that it can mitigate this artifact.
# %% [markdown]
# ### Total-generalized-variation (TGV)
# %% [markdown]
# Similar to the TV case, let $y$ denote the measured data and $A$ the acquisition model. We assume
# $$
# y = Ax_{\mathrm{true}} + n,
# $$
# with (complex) Gaussian noise $n$.
#
# In second-order TGV we introduce an auxiliary field $v$ (with the same spatial shape as $\nabla x$) and minimize
# $$
# \min_{x,\,v}\; \tfrac12\,\lVert Ax - y\rVert_2^2
# \; +\; \lambda_1\,\lVert \nabla x - v\rVert_1
# \; +\; \lambda_0\,\Big\lVert \mathcal{E}v\Big\rVert_1, \tag{1}
# $$
# where $\lambda_1$ and $\lambda_0$ are regularization parameters, and $\mathcal{E}$ is the symmetrized gradient
# $$
# \mathcal{E}v \;=\; \tfrac12\,(\nabla v + (\nabla v)^{\top})
# $$
# %% [markdown]
# #### Recast for PDHG
# To use PDHG we write (1) in the form
# $$
# \min_{x,v}\; g(x,v) + f\big(K(x,v)\big).
# $$
# %% [markdown]
# #### Operator $K$
# $$
# K(x,v) \;=\;
# \begin{bmatrix}
# A & 0\\
# \nabla & -I\\
# 0 & \mathcal{E}
# \end{bmatrix}
# \begin{bmatrix}
# x\\ v
# \end{bmatrix}
# \;=\;
# \begin{bmatrix}
# Ax\\
# \nabla x - v\\
# \mathcal{E}v
# \end{bmatrix}.
# $$
#
# We build this as a `LinearOperatorMatrix` using `FiniteDifferenceOp` (forward/backward modes), a `RearrangeOp` to
# align dimensions, and the simple `ZeroOp`.
# %% [markdown]
# #### Functionals $g$ and $f$
# We take $g \equiv 0$ (implemented implicitly by passing `g=None`), and
# $$
# f(z_1, z_2, z_3) \,=\, \tfrac12\,\lVert z_1 - y\rVert_2^2
# \; +\; \lambda_1\,\lVert z_2\rVert_1
# \; +\; \lambda_0\,\lVert z_3\rVert_1.
# $$
# In code this is
# `ProximableFunctionalSeparableSum(L2NormSquared(y)/2, L1NormViewAsReal(weight=λ1), L1NormViewAsReal(weight=λ0))`.
# We call $\lambda_1$ the gradient term's weight and $\lambda_0$ the symmetrized gradient term's weight.
#
# With this choice of $g$ and $f$, the proximal operators are closed-form.
# %%
def tgv_minimization_reconstruction(
    measurement: torch.Tensor,
    acquisition_op: LinearOperator,
    grad_term_weight: torch.Tensor,
    sym_grad_term_weight: torch.Tensor,
    dim: Sequence[int] = (-2, -1),
    **pdhg_kwargs,
) -> torch.Tensor:
    """Perform TGV-minimization reconstruction."""
    # 1. Compute initial values using the adjoint of the acquisition operator
    adjoint_recon = acquisition_op.adjoint(measurement)[0]
    # Auxiliary tensor v which is in the gradient domain.
    auxiliary_v_tensor = adjoint_recon.new_zeros((len(dim), *adjoint_recon.shape))
    # Increase the number of dimensions of initial image by one.
    # Must make the number of dimensions of the elements in initial values list match one another,
    # because (currently) pdhg function concatenates the individual norms of each operator
    # in a matrix's row and the norm has the same shape as the input.
    # If the number of dimensions don't match, we get a runtime error like this:
    #   RuntimeError: stack expects each tensor to be equal size, but got ... at entry 0 and ... at entry 1
    adjoint_recon = adjoint_recon.unsqueeze(0)
    initial_values = (adjoint_recon, auxiliary_v_tensor)

    # 2. Define the objective functional
    data_term = 0.5 * L2NormSquared(target=measurement)
    grad_term = L1NormViewAsReal(weight=grad_term_weight)
    sym_grad_term = L1NormViewAsReal(weight=sym_grad_term_weight)
    minimization_sum = ProximableFunctionalSeparableSum(data_term, grad_term, sym_grad_term)

    # 3. Define the operator matrix K
    # 3.1. First row (corresponding to the L2-norm data term): Ax + 0
    data_term_row = (acquisition_op, ZeroOp())

    # 3.2. Second row (corresponding to the first L1-norm regularization term): \nabla x - v
    # Reduce the number of dimensions of the image by one before applying the finite difference operator
    # to make the output of the finite difference operator match the auxiliary tensor v.
    squeeze_op = RearrangeOp('1 ... -> ...')
    forward_nabla = FiniteDifferenceOp(dim=dim, mode='forward')
    grad_term_row = (forward_nabla @ squeeze_op, -1 * IdentityOp())

    # 3.3. Third row (corresponding to the second L1-norm regularization term): 0 + \mathcal{E} v
    backward_nabla = FiniteDifferenceOp(dim=dim, mode='backward')
    transpose_op = RearrangeOp('sym_grad_dim  grad_dim  ...   ->   grad_dim  sym_grad_dim  ...')
    symmetric_gradient_op = 0.5 * (1 + transpose_op) @ backward_nabla
    sym_grad_term_row = (ZeroOp(), symmetric_gradient_op)

    operator_matrix = LinearOperatorMatrix((data_term_row, grad_term_row, sym_grad_term_row))

    return mrpro.algorithms.optimizers.pdhg(
        f=minimization_sum,
        g=None,  # automatically converted to zero functionals
        operator=operator_matrix,
        initial_values=initial_values,
        **pdhg_kwargs,
    )[0]


# %% [markdown]
# ### Run PDHG with TGV

# %% [markdown]
# Let us run TGV with weights $(\lambda_1,\lambda_0) = (0.05, 0.1)$; a common heuristic is
# $\lambda_0 \approx 2\lambda_1$. We can later adjust these and the number of iterations.
# %%
square_tgv_denoised = tgv_minimization_reconstruction(
    measurement=square_noisy,
    acquisition_op=identity_op,
    grad_term_weight=torch.tensor(0.05),
    sym_grad_term_weight=torch.tensor(0.1),
    max_iterations=257,
    callback=callback,
)
show_images(
    square_noisy,
    square_tv_denoised,
    square_tgv_denoised,
    square_clean,
    titles=['Noisy', 'TV Denoised', 'TGV Denoised', 'Clean'],
    clim=(0, 1),
)

# %% [markdown]
# Compared to TV, TGV can still preserve edges and details while avoiding staircasing; slowly varying ramps are
# reconstructed more faithfully.
# %% [markdown]
# ## Second example: Radial undersampling
# %% [markdown]
# We next reconstruct a multi-coil brain MRI from retrospectively undersampled radial spokes drawn on a Cartesian grid.
# We use 4-coil k-space data (`1_rawdata_brainT2_4ch.mat`) as a reference dataset
# (fully sampled in Cartesian coordinates).
# %% mystnb={"code_prompt_show": "Show download details"} tags=["hide-cell"]
# Download ground-truth k-space data from Zenodo
zenodo_get.download(record='800525', retry_attempts=5, output_dir=data_folder)

# %% [markdown]
# To begin, we load the fully sampled k-space, form the image with an FFT operator, and compute a
# root-sum-of-squares (RSS) reference.
# %%
from einops import rearrange
from mrpro.data import SpatialDimension
from scipy.io import loadmat

file_name = '1_rawdata_brainT2_4ch.mat'
kdata_true = torch.tensor(loadmat(data_folder / file_name)['rawdata'])
kdata_true = rearrange(kdata_true, 'k1 k0 coils  ->  1 coils 1 k1 k0')
# kdata = torch.flip(kdata, dims=(-2, -1))
kdata_true = kdata_true.to(dtype=torch.complex64)  # If default dtype is torch.float32, use complex64

recon_matrix = SpatialDimension(z=kdata_true.shape[-3], y=kdata_true.shape[-2], x=kdata_true.shape[-1])
encoding_matrix = SpatialDimension(z=kdata_true.shape[-3], y=kdata_true.shape[-2], x=kdata_true.shape[-1])

fourier_op = mrpro.operators.FastFourierOp(
    dim=(-2, -1),
    recon_matrix=recon_matrix,
    encoding_matrix=encoding_matrix,
)
x_true = fourier_op(kdata_true)[0]

x_true_rss = x_true.abs().square().sum(dim=-4).sqrt().squeeze()
show_images(x_true_rss, titles=['Ground Truth'], clim=(0, 7e-4))

# %% [markdown]
# ### Set up the operator $A$
# %% [markdown]
# To set up the acquisition operator $A$, we first create a radial mask with a chosen number of spokes (here 48) on the
# Cartesian grid and wrap it as a `CartesianMaskingOp`.
# %%
from torchvision.transforms.functional import rotate


def radial_mask(ny: int, nx: int, num_spokes: int) -> torch.Tensor:
    """Generate a radial mask with the specified number of spokes."""
    theta = 180 * (3.0 - 5**0.5)  # golden angle ~137.508°
    mask = torch.zeros((1, 1, ny, nx), dtype=torch.bool)

    # prototype spoke: horizontal line through center
    base_spoke = torch.zeros((1, 1, ny, nx))
    base_spoke[0, 0, ny // 2, :] = 1.0

    for i_spoke in range(num_spokes):
        spoke = rotate(base_spoke, angle=theta * i_spoke, fill=0)
        mask |= spoke > 0.5
    return mask


Nx, Ny = 256, 256
num_spokes = 48
mask_op = mrpro.operators.CartesianMaskingOp(radial_mask(Ny, Nx, num_spokes))
show_images(torch.tensor(mask_op.mask), titles=[f'Mask ({num_spokes} spokes)'])

# %% [markdown]
# Then we can define the acquisition operator as
# $$
# A \,=\, M \circ \mathcal{F},
# $$
# where $\mathcal{F}$ is the (multi-dimensional) FFT mapping images to k-space, and $M$ applies the binary mask.
# We simulate undersampling by computing $y = A \ x_{\mathrm{true}}$, and display the adjoint reconstruction $A^* \ y$
# (i.e. zero-filled inverse FFT).
# %%
acquisition_op = mask_op @ fourier_op
# Apply A(x_true) to get undersampled k-space data
kdata_undersampled = acquisition_op(x_true)[0]
x_adjoint_recon = acquisition_op.adjoint(kdata_undersampled)[0]
x_adjoint_recon_rss = torch.sum(x_adjoint_recon.abs() ** 2, dim=1).sqrt()
show_images(x_adjoint_recon_rss, x_true_rss, titles=['Adjoint', 'Ground Truth'], clim=(0, 7e-4))

# %% [markdown]
# ### Run PDHG with TV

# %% [markdown]
# Again, let us start with the TV-regularized reconstruction method. The regularization parameter
# $\lambda = 5 \times 10^{-5}$ was chosen to achieve a good reconstruction for this example.

# %%
x_tv_recon = tv_minimization_reconstruction(
    measurement=kdata_undersampled,
    acquisition_op=acquisition_op,
    grad_term_weight=torch.tensor(5e-6),
    max_iterations=257,
    callback=callback,
)
x_tv_recon_rss = torch.sum(x_tv_recon.abs() ** 2, dim=-4).sqrt()
show_images(x_adjoint_recon_rss, x_tv_recon_rss, x_true_rss, titles=['Adjoint', 'TV', 'Ground Truth'], clim=(0, 7e-4))

# %% [markdown]
# ### Run PDHG with TGV

# %% [markdown]
# Now we can run the TGV-regularized reconstruction with $(\lambda_1, \lambda_0) = (5 \times 10^{-6}, 10^{-5})$.

# %%
x_tgv_recon = tgv_minimization_reconstruction(
    measurement=kdata_undersampled,
    acquisition_op=acquisition_op,
    grad_term_weight=torch.tensor(5e-6),
    sym_grad_term_weight=torch.tensor(10e-6),
    max_iterations=257,
    callback=callback,
)
x_tgv_recon_rss = torch.sum(x_tgv_recon.abs() ** 2, dim=-4).sqrt()
show_images(
    x_adjoint_recon_rss,
    x_tv_recon_rss,
    x_tgv_recon_rss,
    x_true_rss,
    titles=['Adjoint', 'TV', 'TGV', 'Ground Truth'],
    clim=(0, 7e-4),
)

# %% [markdown]
# ### Compare the results
# We compare adjoint (zero-filled), TV, TGV, and the ground truth (RSS) on a zoomed-in portion.
# As expected, adjoint reconstruction shows aliasing from undersampling.
# TV reduces aliasing but introduces some staircasing,
# while TGV is able to reduce aliasing without the staircasing effect.
# %%
show_images(
    x_adjoint_recon_rss[..., :128, :128],
    x_tv_recon_rss[..., :128, :128],
    x_tgv_recon_rss[..., :128, :128],
    x_true_rss[..., :128, :128],
    titles=['Adjoint', 'TV', 'TGV', 'Ground Truth'],
    clim=(0, 7e-4),
)


# %% [markdown]
# That`s it — we performed denoising and MRI reconstruction via TGV-minimization solved with PDHG.
# We can later try different $(\lambda_1,\lambda_2)$ value combinations and adjust the iteration count
# to see how the reconstruction quality changes.
