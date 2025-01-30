# %% [markdown]
# # Total-variation (TV)-minimization reconstruction

# %% [markdown]
# ### Image reconstruction
# Here, we use the Primal Dual Hybrid Gradient (PDHG) algorithm to reconstruct an image from 2D radial k-space data.
#
# Let $y$ denote the k-space data of the image $x_{\mathrm{true}}$ sampled with an acquisition model $A$
# (Fourier transform, coil sensitivity maps, ...), i.e the forward problem is given as
#
# $ y = Ax_{\mathrm{true}} + n, $
#
# where $n$ describes complex Gaussian noise. When using TV-minimization as regularization method, an approximation of
# $x_{\mathrm{true}}$ is obtained by minimizing the following functional $\mathcal{F}$ where $\nabla$ is the
# discretized gradient operator.
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| \nabla x \|_1, \quad \quad \quad (1)$
#
# The minimization of the functional $\mathcal{F}$ is a non-trivial task due to the presence of the operator
# $\nabla$ in the non-differentiable $\ell_1$-norm. A suitable algorithm to solve the problem is the
# PDHG-algorithm [[Chambolle \& Pock, JMIV 2011](https://doi.org/10.1007%2Fs10851-010-0251-1)].\
# PDHG is a method for solving problems of the form
#
# $ \min_x f(K(x)) + g(x)  \quad \quad \quad (2)$
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

dataset = '14617082'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

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
# Before running the TV-minimization reconstruction, we first run a direct (adjoint) reconstruction
# using `~mrpro.algorithms.reconstruction.DirectReconstruction` (see <project:direct_reconstruction.ipynb>)
# of both the 24 spokes and 402 spokes data to have a reference for comparison.

# %%
direct_reconstruction_402 = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_402spokes)
direct_reconstruction_24 = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_24spokes)
img_direct_402 = direct_reconstruction_402(kdata_402spokes)
img_direct_24 = direct_reconstruction_24(kdata_24spokes)

# %% [markdown]
# We also run an iterative SENSE reconstruction (see <project:iterative_sense_reconstruction_radial2D.ipynb>) with early
# stopping of the 24 spokes data. We use it as a comparison and as an initial guess for the TV-minimization
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
# #### $f(z) = f(p,q) = f_1(p) + f_2(q) =  \frac{1}{2}\|p  - y\|_2^2 + \lambda \| q \|_1.$

# %%
regularization_lambda = 0.2
f_1 = 0.5 * mrpro.operators.functionals.L2NormSquared(target=kdata_24spokes.data, divide_by_n=True)
f_2 = regularization_lambda * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=True)
f = mrpro.operators.ProximableFunctionalSeparableSum(f_1, f_2)

# %% [markdown]
# #### $K(x) = [A, \nabla]^T$
#
#   where $\nabla$ is the finite difference operator that computes the directional derivatives along the last two
#   dimensions (y,x), implemented as `~mrpro.operators.FiniteDifferenceOp`, and
#  `~mrpro.operators.LinearOperatorMatrix` can be used to stack the operators.

# %%
nabla = mrpro.operators.FiniteDifferenceOp(dim=(-2, -1), mode='forward')
K = mrpro.operators.LinearOperatorMatrix(((acquisition_operator,), (nabla,)))

# %% [markdown]
# #### $g(x) = 0,$
#
# implemented as `~mrpro.operators.functionals.ZeroFunctional`

# %%
g = mrpro.operators.functionals.ZeroFunctional()
# %% [markdown]
# ```{note}
# An obvious identification would have been
# - $f(x) = \lambda \| x\|_1,$
# - $g(x) = \frac{1}{2}\|Ax  - y\|_2^2,$
# - $K(x) = \nabla x.$
#
# But to be able to compute $\mathrm{prox}_{\tau g}$, one would need to solve a linear system at each
# iteration, making this identification impractical.
# ```

# %% [markdown]
# This identification allows us to compute the proximal operators of $f$ and $g$ easily.
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
        print(f"Iteration {iteration: >3}: Objective = {optimizer_status['objective'](*solution).item():.3e}")


# %%
(img_pdhg_24,) = mrpro.algorithms.optimizers.pdhg(
    f=f,
    g=g,
    operator=K,
    initial_values=(img_sense_24.data,),
    max_iterations=257,
    callback=callback,
)

# %% [markdown]
# ### Compare the results
# We now compare the results of the direct reconstruction, the iterative SENSE reconstruction, and the TV-minimization

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
# Hurrah! We have successfully reconstructed an image from 24 spokes using TV-minimization.
#
# ### Next steps
# Play around with the regularization weight and the number of iterations to see how they affect the final image.
# You can also try to use the 96 spokes data to see how the reconstruction quality improves with more spokes.
