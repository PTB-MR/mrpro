# %% [markdown]
# # Total-generalized-variation (TGV)-minimization reconstruction

# %% [markdown]
# ### Image reconstruction
# Here, we use the Primal Dual Hybrid Gradient (PDHG) algorithm to reconstruct an image from 2D radial k-space data.
#
# Let $y$ denote the k-space data of the image $x_{\mathrm{true}}$ sampled with an acquisition model $A$
# (Fourier transform, coil sensitivity maps, ...), i.e the forward problem is given as
#
# $ y = Ax_{\mathrm{true}} + n, $
#
# where $n$ describes complex Gaussian noise. When using TGV-minimization as regularization method, an approximation of
# $x_{\mathrm{true}}$ is obtained by minimizing the following functional $\mathcal{F}$
#
# $$
# \mathcal{F}(x, v) = \frac{1}{2}||Ax - y||_2^2
# + \lambda_1 \| \nabla x - v \|_1
# + \lambda_0 \| \mathcal{E} v \|_1, \quad \quad \quad (1)
# $$
#
# where $\nabla$ is the discretized gradient operator and $\mathcal{E}$ is the discretized
# symmetrized gradient operator.
#
# The minimization of the functional $\mathcal{F}$ is a non-trivial task due to the presence of the operator
# $\nabla$ in the non-differentiable $\ell_1$-norm. A suitable algorithm to solve the problem is the
# PDHG-algorithm [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)].\
# PDHG is a method for solving problems of the form
#
# $$
# \min_{x, v} f(K(x, v)) + g(x, v)  \quad \quad \quad (2)
# $$
#
# where $f$ and $g$ denote proper, convex, lower-semicontinous functionals and $K$ denotes a linear operator.\
# PDHG essentially consists of three steps, which read as
#
# $z_{k+1} = \mathrm{prox}_{\sigma f^{\ast}}(z_k + \sigma K \bar{x}_k)$
#
# $x_{k+1} = \mathrm{prox}_{\tau g}(x_k - \tau K^H z_{k+1})$
#
# $\bar{x}_{k+1} = x_{k+1} + \theta(x_{k+1} - x_k)$,
#
# where $\mathrm{prox}$ denotes the proximal operator and $f^{\ast}$ denotes the convex conjugate of the
# functional $f$, $\theta\in [0,1]$ and step sizes $\sigma, \tau$ such that $\sigma \tau < 1/L^2$, where
# $L=\|K\|_2$ is the operator norm of the operator $K$.
#
# The first step is to recast problem (1) into the general form of (2) and then to apply the steps above
# in an iterative fashion. In the following, we use this approach to reconstruct a 2D radial image using
# `~mrpro.algorithms.optimizers.pdhg`.

# %% [markdown]
# ### Load data
# Our example data contains three scans acquired with a 2D golden angle radial trajectory and
# varying number of spokes:
#
# - ``radial2D_24spokes_golden_angle_with_traj.h5``
# - ``radial2D_96spokes_golden_angle_with_traj.h5``
# - ``radial2D_402spokes_golden_angle_with_traj.h5``
#
# We will use the 402 spokes as a reference and try to reconstruct the image from the 24 spokes data.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
# Download raw data from Zenodo
import tempfile
from pathlib import Path

import zenodo_get

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.download(record='14617082', retry_attempts=5, output_dir=data_folder)

# %%
import mrpro

# We have embedded the trajectory information in the ISMRMRD files.
kdata_402spokes = mrpro.data.KData.from_file(
    data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5', mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
)
kdata_24spokes = mrpro.data.KData.from_file(
    data_folder / 'radial2D_24spokes_golden_angle_with_traj.h5', mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
)

# %% [markdown]
# ### Comparison reconstructions
# Before running the TGV-minimization reconstruction, we first run a direct (adjoint) reconstruction
# using `~mrpro.algorithms.reconstruction.DirectReconstruction` (see <project:direct_reconstruction.ipynb>)
# of both the 24 spokes and 402 spokes data to have a reference for comparison.

# %%
direct_reconstruction_402 = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_402spokes)
direct_reconstruction_24 = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_24spokes)
img_direct_402 = direct_reconstruction_402(kdata_402spokes)
img_direct_24 = direct_reconstruction_24(kdata_24spokes)

# %% [markdown]
# We also run an iterative SENSE reconstruction (see <project:iterative_sense_reconstruction_radial2D.ipynb>) with early
# stopping of the 24 spokes data. We use it as a comparison and as an initial guess for the TGV-minimization
# reconstruction.

# %%
sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata_24spokes,
    n_iterations=8,
    csm=direct_reconstruction_24.csm,
    dcf=direct_reconstruction_24.dcf,
)
img_sense_24 = sense_reconstruction(kdata_24spokes)

# %% [markdown]
# ### Set up the operator $A$
# Now, to set up the problem, we need to define the acquisition operator $A$, consisting of a
# `~mrpro.operators.FourierOp` and a `~mrpro.operators.SensitivityOp`, which applies the coil sensitivity maps
# (CSM) to the image. We reuse the CSMs estimated in the direct reconstruction.

# %%
fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata_24spokes)

assert direct_reconstruction_24.csm is not None
csm_operator = direct_reconstruction_24.csm.as_operator()

# The acquisition operator is the composition of the Fourier operator and the CSM operator
acquisition_operator = fourier_operator @ csm_operator

# %% [markdown]
# ### Recast the problem to be able to apply PDHG
# To apply the PDHG algorithm, we need to recast the problem into the form of (2). We need to identify
# the functionals $f$ and $g$ and the operator $K$. We chose an identification for which both
# $\mathrm{prox}_{\sigma f^{\ast}}$ and $\mathrm{prox}_{\tau g}$ are easy to compute:
#
# #### $f(z) = f(p,q,r) = f_1(p) + f_2(q) + f_3(r) = \frac{1}{2}\|p - y\|_2^2 + \lambda_1 \|q\|_1 + \lambda_0 \|r\|_1.$

# %%
regularization_lambda_1 = 0.2
regularization_lambda_0 = 0.4
f_1 = 0.5 * mrpro.operators.functionals.L2NormSquared(target=kdata_24spokes.data, divide_by_n=True)
f_2 = regularization_lambda_1 * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=True)
f_3 = regularization_lambda_0 * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=True)
f = mrpro.operators.ProximableFunctionalSeparableSum(f_1, f_2, f_3)

# %% [markdown]
# #### $K = [[A, 0], [\nabla, -I], [0, \mathcal{E}]]^{\top}$
#
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
# \end{bmatrix}
# $$
#
#   where $\nabla$ is the finite difference operator that computes the directional derivatives along the last two
#   dimensions (y,x), implemented as `~mrpro.operators.FiniteDifferenceOp`, and
#   $\mathcal{E}$ is the corresponding symmetrized gradient operator implemented as
#   `~mrpro.operators.SymmetrizedGradientOp`
#
# $$
# \mathcal{E}v \;=\; \tfrac12\,(\nabla v + (\nabla v)^{\top})
# $$
#
#  `~mrpro.operators.LinearOperatorMatrix` can be used to stack the operators.

# %%
# Directions along which to compute the finite differences
dim = (-2, -1)

# Define the operator matrix K
# 1. First row [A, 0] (corresponding to the L2-norm data term): Ax + 0
data_term_row = (acquisition_operator, mrpro.operators.ZeroOp())

# 2. Second row [\nabla, -I] (corresponding to the first L1-norm regularization term): \nabla x - v
# Reduce the number of dimensions of the image by one before applying the finite difference operator
# to make the output of the finite difference operator match the auxiliary tensor v.
squeeze_op = mrpro.operators.RearrangeOp('1 ... -> ...')
forward_nabla = mrpro.operators.FiniteDifferenceOp(dim=dim, mode='forward')
grad_term_row = (forward_nabla @ squeeze_op, -1 * mrpro.operators.IdentityOp())

# 3. Third row [0, \mathcal{E}] (corresponding to the second L1-norm regularization term): 0 + \mathcal{E} v
symmetric_gradient_op = mrpro.operators.SymmetrizedGradientOp(dim=dim, mode='backward')
sym_grad_term_row = (mrpro.operators.ZeroOp(), symmetric_gradient_op)

K = mrpro.operators.LinearOperatorMatrix((data_term_row, grad_term_row, sym_grad_term_row))

# %% [markdown]
# #### $g(x, v) = 0,$
#
# implemented as `~mrpro.operators.functionals.ZeroFunctional`

# %%
g = ProximableFunctionalSeparableSum = mrpro.operators.ProximableFunctionalSeparableSum(
    *(mrpro.operators.functionals.ZeroFunctional(),) * 2
)


# %% [markdown]
# ### Run PDHG for a certain number of iterations
# Now we can run the PDHG algorithm to solve the minimization problem. We use
# the iterative SENSE image as an initial value to speed up the convergence.
# ```{note}
# We can use the `callback` parameter of `~mrpro.algorithms.optimizers` to get some information
# about the progress. In the collapsed cell, we implement a simple callback function that print the status
# message
# ```

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show callback details"}
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
# Auxiliary tensor v which is in the gradient domain.
auxiliary_v_tensor = img_sense_24.data.new_zeros((len(dim), *img_sense_24.shape))
# Increase the number of dimensions of initial image by one.
# Must make the number of dimensions of the elements in initial values list match one another,
# because (currently) pdhg function concatenates the individual norms of each operator
# in a matrix's row and the norm has the same shape as the input.
# If the number of dimensions don't match, we get a runtime error like this:
#   RuntimeError: stack expects each tensor to be equal size, but got ... at entry 0 and ... at entry 1
initial_image = img_sense_24.data.unsqueeze(0)

(img_pdhg_24, _) = mrpro.algorithms.optimizers.pdhg(
    f=f,
    g=g,
    operator=K,
    initial_values=(initial_image, auxiliary_v_tensor),
    max_iterations=257,
    callback=callback,
)

# %% [markdown]
# ### Compare the results
# We now compare the results of the direct reconstruction, the iterative SENSE reconstruction, and the TGV-minimization

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
import matplotlib.pyplot as plt
import torch


def show_images(*images: torch.Tensor, titles: list[str] | None = None) -> None:
    """Plot images."""
    n_images = len(images)
    _, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 3, 3))
    for i in range(n_images):
        axes[0][i].imshow(images[i], cmap='gray')
        axes[0][i].axis('off')
        if titles:
            axes[0][i].set_title(titles[i])
    plt.show()


# %%
# see the collapsed cell above for the implementation of show_images
show_images(
    img_direct_402.rss().squeeze(),
    img_direct_24.rss().squeeze(),
    img_sense_24.rss().squeeze(),
    img_pdhg_24.abs().squeeze(),
    titles=['402 spokes', '24 spokes (direct)', '24 spokes (SENSE)', '24 spokes (PDHG)'],
)

# %% [markdown]
# Hurrah! We have successfully reconstructed an image from 24 spokes using TGV-minimization.
#
# ### Next steps
# Play around with the regularization weight and the number of iterations to see how they affect the final image.
# You can also try to use the 96 spokes data to see how the reconstruction quality improves with more spokes.
