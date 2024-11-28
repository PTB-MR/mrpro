# %% [markdown]
# # Total-variation (TV)-minimization reconstruction of 2D golden angle radial data

# %%
# define zenodo URL of the example ismrmd data
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'

# %% [markdown]
# ##### Download and read-in the raw data

import tempfile

import mrpro
import requests

# Download the data from zenodo
data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.h5')
response = requests.get(zenodo_url + fname, timeout=30)
data_file.write(response.content)
data_file.flush()

# Load in the data from the ISMRMRD file
kdata = mrpro.data.KData.from_file(data_file.name, mrpro.data.traj_calculators.KTrajectoryIsmrmrd())
kdata.header.recon_matrix.x = 256
kdata.header.recon_matrix.y = 256


# %% [markdown]
# ### Image reconstruction
# Here, we use the Primal Dual Hybrid Gradient (PDHG) algorithm to reconstruct an image from 2D radial k-space data
# data.
#
# Let $y$ denote the k-space data of the image $x_{\mathrm{true}}$ sampled with an acquisition model $A$
# (Fourier transform, coil sensitivity maps, ...), i.e the forward problem is given as
#
# $ y = Ax_{\mathrm{true}} + n, $
#
# where $n$ describes complex Gaussian noise. When using TV-minimization as regularization method, an approximation of
# $x_{\mathrm{true}}$ is obtained by minimizing the following functional $\mathcal{F}$:
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| \nabla x \|_1, \quad \quad \quad (1)$
#
# where $\nabla$ is the discretized gradient operator.
#
# The minimization of the functional $\mathcal{F}$ is a non-trivial task due to the presence of the operator
# $\nabla$ in the non-differentiable $\ell_1$-norm. A suitable algorithm to solve the problem is the
# PDHG-algorithm [Chambolle \& Pock, JMIV 2011].
#
# PDHG is a method for solving problems of the form
#
# $ \min_x f(K(x)) + g(x)  \quad \quad \quad (2)$
#
# where $f$ and $g$ denote proper, convex, lower-semicontinous functionals and $K$ denotes a linear operator.
#
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
# in an iterative fashion.
#
# A possible and intuitive (but unfortunately not efficient) identification is the following
#
# $f(x) = \lambda \| x\|_1,$
#
# $g(x) = \frac{1}{2}\|Ax  - y\|_2^2,$
#
# $K(x) = \nabla x.$
#
# However, although $\mathrm{prox}_{\sigma f^\ast}$ has a simple form, calculations show that
# to be able to compute $\mathrm{prox}_{\tau g}$, one would need to solve a linear system at each
# iteration. We will therefore use a way more efficient identification.
#
# In the following, we load 2D radial MR data and set up problem (2) to use PDHG to reconstruct the data.

# %% [markdown]
# ### Set up the operator $A$
# Estimate coil sensitivity maps and density compensation function. Also run a direct (adjoint)
# reconstruction and iterative SENSE as methods of comparison.

# %%
# Set up direct reconstruction class. The estimate coil sensitivity maps and density
# compensation values can be reused later to save time.
direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_direct = direct_reconstruction(kdata)

# run iterative SENSE to get an initial guess of the solution
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata,
    n_iterations=8,
    csm=direct_reconstruction.csm,
    dcf=direct_reconstruction.dcf,
)
img_iterative_sense = iterative_sense_reconstruction(kdata)

# Define Fourier operator and CSM operator re-using the ones already constructed in the
# direct reconstruction class
fourier_operator = direct_reconstruction.fourier_op
assert direct_reconstruction.csm is not None
csm_operator = direct_reconstruction.csm.as_operator()

# Create the entire acquisition operator A
acquisition_operator = fourier_operator @ csm_operator

# %% [markdown]
# ### Recast the problem to be able to apply PDHG
# As mentioned, the previously described identification is not efficient.
# Another (less intuitive, but way more efficient) identification is the following:
#
# $f(z) = f(p,q) = f_1(p) + f_2(q) =  \frac{1}{2}\|p  - y\|_2^2 + \lambda \| q \|_1,$
#
# $K(x) = [A, \nabla]^T,$
#
# $g(x) = 0,$
#
# for which one can show that both $\mathrm{prox}_{\sigma f^{\ast}}$ and $\mathrm{prox}_{\tau g}$ are
# given by simple and easy-to-compute operations, see for example [Sidky et al, PMB 2012].
#
# In the following, we first identify the functionals $f$, $g$ and the operator $K$ and then run PDHG.

# %%
# Define the gradient operator \nabla to be used in the operator K=[A, \nabla]^T for PDHG
from mrpro.operators import FiniteDifferenceOp

# The operator computes the directional derivatives along the the last two dimensions (x,y)
nabla_operator = FiniteDifferenceOp(dim=(-2, -1), mode='forward')

from mrpro.algorithms.optimizers import pdhg
from mrpro.algorithms.optimizers.pdhg import PDHGStatus
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional

# Regularization parameter for the $\ell_1$-norm
regularization_parameter = 0.1

# Set up the problem by using the previously described identification
l2 = 0.5 * L2NormSquared(target=kdata.data, divide_by_n=True)
l1 = regularization_parameter * L1NormViewAsReal(divide_by_n=True)

f = ProximableFunctionalSeparableSum(l2, l1)
g = ZeroFunctional()
K = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

# initialize PDHG with iterative SENSE solution for warm start
initial_values = (img_iterative_sense.data,)

# %% [markdown]
# ### Run PDHG for a certain number of iterations

# %%
max_iterations = 128


# call backfunction to track the value of the objective functional f(K(x)) + g(x)
def callback(optimizer_status: PDHGStatus) -> None:
    """Print the value of the objective functional every 8th iteration."""
    iteration = optimizer_status['iteration_number']
    solution = optimizer_status['solution']
    if iteration % 16 == 0:
        print(optimizer_status['objective'](*solution).item())


(img_pdhg,) = pdhg(
    f=f, g=g, operator=K, initial_values=initial_values, max_iterations=max_iterations, callback=callback
)

# %%
# ### Compare the results
# Display the reconstructed images
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, squeeze=False)
clim = [0, 1e-3]
ax[0, 0].set_title('Adjoint (direct) Recon', fontsize=10)
ax[0, 0].imshow(img_direct.data.abs()[0, 0, 0, :, :], clim=clim)
ax[0, 1].set_title('Iterative SENSE', fontsize=10)
ax[0, 1].imshow(img_iterative_sense.data.abs()[0, 0, 0, :, :], clim=clim)
ax[0, 2].set_title('PDHG', fontsize=10)
ax[0, 2].imshow(img_pdhg.abs()[0, 0, 0, :, :], clim=clim)
plt.setp(ax, xticks=[], yticks=[])

# %% [markdown]
# ### Next steps
# Play around with the regularization weight and the number of iterations to see how they effect the final image.
