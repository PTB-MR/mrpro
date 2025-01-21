# %% [markdown]
# # Proximal gradient descent (FISTA) reconstruction of 2D golden angle radial data

# %% [markdown]
# ##### Download and read-in the raw data

# %%
import tempfile

import mrpro
import requests

# define zenodo URL of the example ismrmd data
# choose number of spokes; either 24, 96402
n_spokes = 24
zenodo_url = 'https://zenodo.org/records/14617082/files/'
fname = f'radial2D_{n_spokes}spokes_golden_angle_with_traj.h5'

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
# Here, we use a proximal gradient descent algorithm to reconstruct an image
# from 2D radial k-space data with wavelet regularization.
# In particular, we use the accelerated version known as FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
# [Beck \& Teboulle, SIAM Journal on Imaging Sciences 2009].
#
# Let $y$ denote the k-space data of the image $x_{\mathrm{true}}$ sampled with an acquisition model $A$
# (Fourier transform, coil sensitivity maps, etc.), i.e the forward problem is given as
#
# $ y = Ax_{\mathrm{true}} + n, $
#
# where $n$ describes complex Gaussian noise.
# An approximation of $x_{\mathrm{true}}$ can be carried out exploiting
# the fact that wavelets provide a sparse representation of images
# by minimizing
# the following functional $\mathcal{F}$:
#
# $ \mathcal{F}(x) = \frac{1}{2}||Ax - y||_2^2 + \lambda \| W x \|_1, \quad \quad \quad (1)$
#
# where $W$ is the discretized wavelet operator that maps image domain into wavelet domain,
# and $\lambda >0$ appropriate weight.
#
# The minimization of the functional $\mathcal{F}$ is non-trivial due to
# the non-differentiable $\ell_1$-norm and the presence of the wavelet operator.
# We proceed first by reformulating $(1)$ to solve the minimization problem in the wavelet domain,
# exploiting the orthonormality of $W$.
# We set a new variable $\tilde{x} = Wx$. Thus,
# the minimization problem becomes
#
# $ \min_{\tilde{x}} \frac{1}{2}||AW^H\tilde{x} - y||_2^2 + \lambda \| \tilde{x} \|_1.  \quad \quad \quad (2)$
#
# A suitable algorithm to solve $(2)$ is the
# FISTA-algorithm, consisting in an accelerated proximal gradient descent algorithm.
# It solves problems of the form
#
# $ \min_x f(x) + g(x)  \quad \quad \quad (3)$
#
# where $f$ is a convex, differentiable function with $L$-Lipschitz gradient $\nabla f$,
# and $g$ is convex and possibly non-smooth.
#
# The main step of the minimization method is a proximal gradient step, which reads as
#
# $x_{k} = \mathrm{prox}_{\sigma g}(x_{k-1} - \sigma \nabla f({x}_{k-1}))$
#
# where $\mathrm{prox}$ denotes the proximal operator and $\sigma$ is
# an appropriate stepsize, ideally $\sigma=\frac{1}{L}$, with $L$ being
# the Lipschitz constant of $\nabla f$.
#
# In FISTA, there is an
# additional step to accelerate the convergence. The following variable $y_{k+1}$
# is added, consisting of a linear interpolation
# of the previous two steps $x_{k}$ and $x_{k-1}$. So, for $t_1=1, y_1=x_0$:
#
# $x_{k} = \mathrm{prox}_{\sigma g}(y_{k} - \sigma \nabla f({y}_{k}))$
#
# $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$
#
# $y_{k+1} = x_{k} + \frac{t_k - 1}{t_{k+1}}(x_{k} - x_{k-1}).$
#
# As the Lipschitz constant $L$ is in general not known, and
# the interval of the stepsize $\sigma\in ( 0, \frac{1}{|| W A^H A W^H||} )$ is crucial for the convergence,
# a backtracking step can
# be performed to update the stepsize $\sigma$ at every iteration. To do so,
# $\sigma$ is iteratively reduced until reaching a stepsize that is the biggest one
# for which the quadratic approximation of $f$ at $y_{k}$
# is an upper bound for $f(x_{k})$.
#
# Functions $f$ ang $g$ from $(3)$ can be identified as
#
# $f(x) = \frac{1}{2}\|AW^Hx  - y\|_2^2,$
#
# $g(x) = \lambda \| x\|_1.$
#
# After running the algorithm for $N$ iterations, the optimal solution $x_{N}$
# is in wavelet domain and needs to be mapped back to image domain.
# Thus we apply the adjoint of the wavelet transform and obtain solution $x_{\text{opt}}$ as
#
# $x_{\text{opt}} := W^H x_{N}$.
#
# In the following, we load 2D radial MR data and set up problem (2) to use
# FISTA to reconstruct the data.

# %% [markdown]
# ### Set up the operator $A$
# Estimate coil sensitivity maps and density compensation function. Also run a direct (adjoint)
# reconstruction and iterative SENSE as methods of comparison.
#
# Define wavelet operator $W$ and set $A = F C W^H $, where $F$ is the Fourier
# operator, $C$ the coil sensitivity maps and $W^H$ the adjoint wavelet operator.

# %%
# Set up direct reconstruction class. 
direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
img_direct = direct_reconstruction(kdata)

# run iterative SENSE to compare the solution
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

# Define the wavelet operator
wavelet_operator = mrpro.operators.WaveletOp(domain_shape=img_direct.data.shape[-2:], dim=(-2, -1))

# Create the full acquisition operator A with wavelet operator
acquisition_operator = fourier_operator @ csm_operator @ wavelet_operator.H

# %% [markdown]
# ### Apply FISTA

# In the following, we first identify the functionals $f$ and $g$
# and then run FISTA.

# %%
from mrpro.algorithms.optimizers import pgd
from mrpro.algorithms.optimizers.pgd import PGDStatus
from mrpro.operators.functionals import L1Norm, L2NormSquared

# Regularization parameter for the $\ell_1$-norm
regularization_parameter = 1e-6

# Set up the problem by using the previously described identification
l2 = 0.5 * L2NormSquared(target=kdata.data, divide_by_n=False)
l1 = L1Norm(divide_by_n=False)

f = l2 @ acquisition_operator
g = regularization_parameter * l1

# initialize FISTA with adjoint solution
initial_values = wavelet_operator(img_direct.data)

# %% [markdown]
# ### Run FISTA for a certain number of iterations

max_iterations = 25

# callback function to track the value of the objective functional f(x) + g(x)
# and stepsize update
def callback(optimizer_status: PGDStatus) -> None:
    """Print the value of the objective functional every 10th iteration."""
    iteration = optimizer_status['iteration_number']
    solution = optimizer_status['solution']
    if iteration % 10 == 0:
        print(
            f'{iteration}: {optimizer_status["objective"](*solution).item()}, stepsize: {optimizer_status["stepsize"]}'
        )

(img_wave_pgd,) = pgd(
    f=f,
    g=g,
    initial_value=initial_values,
    stepsize=1,
    max_iterations=max_iterations,
    backtrack_factor=0.85,
    callback=callback,
)

# map the solution back to image domain
(img_pgd,) = wavelet_operator.H(img_wave_pgd)


# %%
# ### Compare the results
# Display the reconstructed images
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
clim = [0, 1e-3]
ax[0].set_title('Adjoint (direct) Recon', fontsize=10)
ax[0].imshow(img_direct.data.abs()[0, 0, 0, :, :], clim=clim)
ax[1].set_title('Iterative SENSE', fontsize=10)
ax[1].imshow(img_iterative_sense.data.abs()[0, 0, 0, :, :], clim=clim)
ax[2].set_title('FISTA', fontsize=10)
ax[2].imshow(img_pgd.abs()[0, 0, 0, :, :], clim=clim)
plt.setp(ax, xticks=[], yticks=[])

plt.show()
# %%
