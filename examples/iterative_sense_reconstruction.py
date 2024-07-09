# %% [markdown]
# # Iterative SENSE Reconstruction of 2D golden angle radial data
# Here we use the IterativeSenseReconstruction class to reconstruct images from ISMRMRD 2D radial data
# %%
# define zenodo URL of the example ismrmd data
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
# %%
# Download raw data
import tempfile
from pathlib import Path

import requests

data_folder = Path(tempfile.mkdtemp())
data_file = tempfile.NamedTemporaryFile(dir=data_folder, mode='wb', delete=False, suffix='.h5')
response = requests.get(zenodo_url + fname, timeout=30)
data_file.write(response.content)

# %% [markdown]
# ### Image reconstruction
# We use the IterativeSenseReconstruction class to reconstruct images from 2D radial data.
# IterativeSenseReconstruction solves the following reconstruction problem:
#
# Let's assume we have obtained the k-space data $y$ from an image $x$ with an acquisition model (Fourier transforms,
# coil sensitivity maps...) $A$ then we can formulate the forward problem as:
#
# $ y = Ax + n $
#
# where $n$ describes complex Gaussian noise. Now we want to solve the inverse problem by minimizing
#
# $ \min_x \frac{1}{2}||W^{\frac{1}{2}}(Ax - y)||_2^2 $
#
# where $W^\frac{1}{2}$ is the square root of the density compensation function. We can rewrite this problem as:
#
# $ W^\frac{1}{2}Ax = W^\frac{1}{2}y$
#
# $ A^H W A x = A^H W y$
#
# $ H x = b $ $\quad$ Eq (1)
#
# with $H = A^H W A$ and $b = A^H W y$ which can be solved with a conjugate gradient approach.
# %%
import mrpro

# Use the trajectory that is stored in the ISMRMRD file
trajectory = mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
# Load in the Data from the ISMRMRD file
kdata = mrpro.data.KData.from_file(data_file.name, trajectory)
kdata.header.recon_matrix.x = 256
kdata.header.recon_matrix.y = 256

# %%
iterative_sense_reconstruction = mrpro.algorithms.reconstruction.IterativeSenseReconstruction.from_kdata(
    kdata, n_iterations=4
)
img = iterative_sense_reconstruction(kdata)


# %% [markdown]
# ### Behind the scenes

# %% [markdown]
# ##### $W$

# %%
# Calculate dcf using the trajectory
dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()


# %% [markdown]
# ##### $A$

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
# ##### $b = A^H W y$

# %%
(right_hand_side,) = acquisition_operator.H(dcf_operator(kdata.data)[0])


# %% [markdown]
# ##### $H = A^H W A$

# %%
cg_operator = acquisition_operator.H @ dcf_operator @ acquisition_operator

# %% [markdown]
# ##### Conjugate gradient minimisation

# %%
import torch

img_manual = mrpro.algorithms.optimizers.cg(
    cg_operator, right_hand_side, initial_value=right_hand_side, max_iterations=4, tolerance=0.0
)


# %%
# For comparison we can also carry out a direct reconstruction
direct_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction.from_kdata(kdata)
img_direct = direct_reconstruction(kdata).rss().cpu()
img_direct = img_direct

# %%
# Display the reconstructed image
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, squeeze=False)
ax[0, 0].imshow(img_direct.data[0, 0, :, :])
ax[0, 0].set_title('Direct Reconstruction', fontsize=10)
ax[0, 1].imshow(torch.abs(img.data[0, 0, 0, :, :]))
ax[0, 1].set_title('Iterative SENSE', fontsize=10)
ax[0, 2].imshow(torch.abs(img_manual[0, 0, 0, :, :]))
ax[0, 2].set_title('"Manual" Iterative SENSE', fontsize=10)

# %% [markdown]
# ### Check for equal results
# The two versions result should in the same image data.

# %%
# If the assert statement did not raise an exception, the results are equal.
assert torch.allclose(img.data, img_manual)

# %% [markdown]
# Copyright 2024 Physikalisch-Technische Bundesanstalt
# Apache License 2.0. See LICENSE file for details.
