# %% [markdown]
# # Total-Variation (TV)-minimization Reconstruction of 2D golden angle radial data
# Here we use the PDHG algorithm to reconstruct images from ISMRMRD 2D radial data
# %%
# define zenodo URL of the example ismrmd data
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
# %%
# Download raw data
import tempfile

import requests

data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.h5')
response = requests.get(zenodo_url + fname, timeout=30)
data_file.write(response.content)
data_file.flush()

# %% [markdown]
# ### Image reconstruction
# We use the Primal Dual Hybrid Gradient (PDHG) algorithm to reconstruct images from 2D radial data.
#
# Let's assume we have obtained the k-space data $y$ from an image $x$ with an acquisition model (Fourier transforms,
# coil sensitivity maps...) $A$ then we can formulate the forward problem as:
#
# $ y = Ax + n $
#
# where $n$ describes complex Gaussian noise. When using TV-minimization as regularization method, the image $x$ can be
# obtained by minimizing the following functional $\mathcal{F}$
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| \nabla x \|_1, \quad \quad \quad (1)$
#
# where $W^\frac{1}{2}$ is the square root of the density compensation function (which corresponds to a diagonal
# operator) and $\nabla$ is the discretized gradient operator.
#
# The minimization of the functional $\mathcal{F}$ is a non-trivial task due to the presence of the operator
# $\nabla$ in the non-differentiable $\ell_1$-norm. A suitable algorithm to solve the problem is the
# PDHG-algorithm [Chambolle \& Pock, JMIV 2011].
#
# PDHG is a method for solving the problem
#
# $ \min_x f(K(x)) + g(x)  \quad \quad \quad (2)$
#
# where $f$ and $g$ denote proper. convex, lower-semicontinous functionals and $K$ denotes a linear operator.
#
# PDHG then, essentially consists of three steps, which read as
#
# $z_{k+1} = \mathrm{prox}_{\sigma f^{\ast}}(z_k + \sigma K \bar{x}_k)$ \n
# $x_{k+1} = \mathrm{prox}_{\tau g}(x_k - \tau K^H z_{k+1})$ \n
# $\bar{x}_{k+1} = x_{k+1} + \theta(x_{k+1} - x_k)$,
#
# where $\mathrm{prox}$ denotes the proximal operator and $f^{\ast}$ denotes the convex conjugate of the
# functional $f$, $\theta\in [0,1]$ and step sizes $\sigma, \tau$ such that $\sigma \tau < 1/L^2$, where $L$ is
# the operator norm of the operator $K$.
#
# The first step is to recast problem (1) into the general form of (2) and then to apply the steps above
# in an iterative fashion.
#
# An possible and intuitive (but unfortunately not efficient) identification is the following
#
# $f(x) = \lambda \| x\|_1,$\n
# $g(x) = \frac{1}{2}\|Ax  - y\|_2^2,$\n
# $K(x) = \nabla x$,
#
# However, although $\mathrm{prox}_{\sigma f^\ast}$ has a simple form, some calculations show that
# to be able compute the $\mathrm{prox}_{\tau g}$, one would need to solve a linear system at each
# iteration.
#
# Thus, another (less intuitive, but way more efficient) identification is the following:
#
# $f(z) = f(p,q) = f_1(p) + f_2(q) =  \frac{1}{2}\|p  - y\|_2^2 + \lambda \| q \|_1,$\n
# $K(x) = [A, \nabla]^T,$\n
# $g(x) = 0,$
#
# for which, one can show that both $\mathrm{prox}_{\sigma f^{\ast}}$ and $\mathrm{prox}_{\tau g}$ are
# given by simple easy-to-compute operations, see for example [Sidky et al, PMB 2012].
#
# In the following, we load some 2D radial MR data and use the just described identification to set up
# the corresponding problem to be solved with PDHG.

# %%
import mrpro

# %% [markdown]
# ##### Read-in the raw data

# %%
# Use the trajectory that is stored in the ISMRMRD file
trajectory = mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
# Load in the Data from the ISMRMRD file
kdata = mrpro.data.KData.from_file(data_file.name, trajectory)
kdata.header.recon_matrix.x = 256
kdata.header.recon_matrix.y = 256

# %% [markdown]
# ### Set up the operator$A$
# Estimate coil sensitivity maps and density compensation function. Also run a direct (adjoint)
# reconstruction and iterative SENSE as methods of comparison.

# %%
# Set up direct reconstruction class to estimate coil sensitivity maps, density compensation etc
direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_direct = direct_reconstruction(kdata).data

# run tierative SENSE to get an initiual guess of the solution
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata, n_iterations=8, csm=direct_reconstruction.csm
)
img_iterative_sense = iterative_sense_reconstruction(kdata).data

# Define Fourier operator using the trajectory and header information in kdata
fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata)

# estimate density compensation
dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()

# Calculate coil maps
img_coilwise = mrpro.data.IData.from_tensor_and_kheader(*fourier_operator.H(*dcf_operator(kdata.data)), kdata.header)
csm_operator = mrpro.data.CsmData.from_idata_walsh(img_coilwise).as_operator()

# Create the entire acquisition operator A
acquisition_operator = fourier_operator @ csm_operator

# adjoint and iterative SENSE reconstruction as comparison methods
# adjoint reconstruction
(img_adjoint,) = acquisition_operator.H(kdata.data)


# %% [markdown]
# ### Recast the problem for PDHG to be applicable

# %%
# Define the gradient operator
from mrpro.operators import FiniteDifferenceOp

# The operator computes the directional derivatives along the penultimate and last dimensions (x,y)
nabla_operator = FiniteDifferenceOp(dim=(-2, -1), mode='forward')

# Set up the problem by using the previously described identification
from mrpro.algorithms.optimizers import pdhg
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional

# Regularization parameter for the $\ell_1$-norm
regularization_parameter = 0.5

# Define the separable functionals
kdata_tensor = kdata.data
l2 = 0.5 * L2NormSquared(target=kdata_tensor, divide_by_n=True)
l1 = regularization_parameter * L1NormViewAsReal(divide_by_n=True)

# Define the functionals f and g
f = ProximableFunctionalSeparableSum(l2, l1)
g = ZeroFunctional()
operator = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

# initialize PDHG with iterative SENSE solution for warm start
initial_values = (img_iterative_sense.data,)

# %% [markdown]
# ### Run PDHG for a certain number of iterations
max_iterations = 32
(img_pdhg,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=max_iterations)


# %%
# ### Compare the results
# Display the reconstructed images
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, squeeze=False)
clim = [0, 1e-3]
ax[0, 0].set_title('Adjoint Recon', fontsize=10)
ax[0, 0].imshow(img_direct.abs()[0, 0, 0, :, :], clim=clim)
ax[0, 1].set_title('Iterative SENSE', fontsize=10)
ax[0, 1].imshow(img_iterative_sense.data.abs()[0, 0, 0, :, :], clim=clim)
ax[0, 2].set_title('PDHG', fontsize=10)
ax[0, 2].imshow(img_pdhg.abs()[0, 0, 0, :, :], clim=clim)
plt.setp(ax, xticks=[], yticks=[])

# %%
