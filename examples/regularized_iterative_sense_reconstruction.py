# %% [markdown]
# # Regularized Iterative SENSE Reconstruction of 2D golden angle radial data
# Here we use the RegularizedIterativeSENSEReconstruction class to reconstruct images from ISMRMRD 2D radial data
# %%
# define zenodo URL of the example ismrmd data
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
# %%
# Download raw data
import tempfile

import requests

#data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.h5')
#response = requests.get(zenodo_url + fname, timeout=30)
#data_file.write(response.content)
#data_file.flush()

# %% [markdown]
# ### Image reconstruction
# We use the RegularizedIterativeSENSEReconstruction class to reconstruct images from 2D radial data.
# RegularizedIterativeSENSEReconstruction solves the following reconstruction problem:
#
# Let's assume we have obtained the k-space data $y$ from an image $x$ with an acquisition model (Fourier transforms,
# coil sensitivity maps...) $A$ then we can formulate the forward problem as:
#
# $ y = Ax + n $
#
# where $n$ describes complex Gaussian noise. The image $x$ can be obtained by minimising the functionl $F$
#
# $ F(x) = ||W^{\frac{1}{2}}(Ax - y)||_2^2 $
#
# where $W^\frac{1}{2}$ is the square root of the density compensation function (which corresponds to a diagonal
# operator). Because this is an ill-posed problem, we can add a regularisation term to stablize the problem and obtain
# a solution with certain properties:
#
# $ F(x) = ||W^{\frac{1}{2}}(Ax - y)||_2^2 + l||Bx - x_{reg}||_2^2$
#
# where $l$ is the strength of the regularisation, $B$ is a linear operator and $x_{reg}$ is a regularisation image.
# With this functional $F$ we obtain a solution which is close to $x_{reg}$ and to the acquired data $y$.
#
# Setting the derivative of the functional $F$ to zero and rearranging yields
#
# $ (A^H W A + l B) x = A^H W y + l x_{reg}$
#
# which is a linear system $Hx = b$ that needs to be solved for $x$.
# %%
import mrpro

# %% [markdown]
# ##### Read-in the raw data

fname = '/Users/kolbit01/Documents/PTB/Data/mrpro/raw/pulseq_radial_2D_24spokes_golden_angle_with_traj.h5'

# %%
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd

# Load in the Data and the trajectory from the ISMRMRD file
kdata = KData.from_file(fname, KTrajectoryIsmrmrd())
kdata.header.recon_matrix.x = 256
kdata.header.recon_matrix.y = 256

# %% [markdown]
# ##### Direct reconstruction and Iterative SENSE reconstruction for comparison

# %%
from mrpro.algorithms.reconstruction import DirectReconstruction, IterativeSENSEReconstruction

# For comparison we can carry out a direct reconstruction
direct_reconstruction = DirectReconstruction(kdata)
img_direct = direct_reconstruction(kdata)

iterative_sense_reconstruction = IterativeSENSEReconstruction(kdata, csm=direct_reconstruction.csm, n_iterations=6)
img_iterative_sense = iterative_sense_reconstruction(kdata)

# %%
import matplotlib.pyplot as plt

vis_im = [img_direct.rss(), img_iterative_sense.rss()]
vis_title = ['Direct', 'Iterative SENSE']
fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(8, 4))
for ind in range(2):
    ax[0, ind].imshow(vis_im[ind][0, 0, ...])
    ax[0, ind].set_title(vis_title[ind])

# %% [markdown]
# ##### Regularized Iterative SENSE reconstruction

# %%
# We can use the direct reconstruction to obtain the coil maps.
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata, csm=direct_reconstruction.csm, n_iterations=4
)
img = iterative_sense_reconstruction(kdata)

# %% [markdown]
# ### Behind the scenes

# %% [markdown]
# ##### Set-up the density compensation operator $W$

# %%
# The density compensation operator is calculated based on the k-space locations of the acquired data.
dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()


# %% [markdown]
# ##### Set-up the acquisition model $A$

# %%
# Define Fourier operator using the trajectory and header information in kdata
fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata)

# Calculate coil maps
# Note that operators return a tuple of tensors, so we need to unpack it,
# even though there is only one tensor returned from adjoint operator.
img_coilwise = mrpro.data.IData.from_tensor_and_kheader(*fourier_operator.H(*dcf_operator(kdata.data)), kdata.header)
csm_operator = mrpro.data.CsmData.from_idata_walsh(img_coilwise).as_operator()

# Create the acquisition operator A
acquisition_operator = fourier_operator @ csm_operator

# %% [markdown]
# ##### Calculate the right-hand-side of the linear system $b = A^H W y$

# %%
(right_hand_side,) = acquisition_operator.H(dcf_operator(kdata.data)[0])


# %% [markdown]
# ##### Set-up the linear self-adjoint operator $H = A^H W A$

# %%
operator = acquisition_operator.H @ dcf_operator @ acquisition_operator

# %% [markdown]
# ##### Run conjugate gradient

# %%
img_manual = mrpro.algorithms.optimizers.cg(
    operator, right_hand_side, initial_value=right_hand_side, max_iterations=4, tolerance=0.0
)

# %%
# Display the reconstructed image
import matplotlib.pyplot as plt
import torch

fig, ax = plt.subplots(1, 3, squeeze=False)
ax[0, 0].imshow(img_direct.rss()[0, 0, :, :])
ax[0, 0].set_title('Direct Reconstruction', fontsize=10)
ax[0, 1].imshow(img.rss()[0, 0, :, :])
ax[0, 1].set_title('Iterative SENSE', fontsize=10)
ax[0, 2].imshow(img_manual.abs()[0, 0, 0, :, :])
ax[0, 2].set_title('"Manual" Iterative SENSE', fontsize=10)

# %% [markdown]
# ### Check for equal results
# The two versions result should in the same image data.

# %%
# If the assert statement did not raise an exception, the results are equal.
assert torch.allclose(img.data, img_manual)
