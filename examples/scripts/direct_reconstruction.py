# %% [markdown]
# # Direct Reconstruction of 2D golden angle radial data
# Here we use the DirectReconstruction class to reconstruct images from ISMRMRD 2D radial data
# %% tags=["hide-cell"]
# Download raw data from Zenodo
import tempfile
from pathlib import Path

import zenodo_get

dataset = '14617082'

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %% [markdown]
# ## Image reconstruction
# We use the `mrpro.algorithms.reconstruction.DirectReconstruction` class to reconstruct images from 2D radial data.
# `~mrpro.algorithms.reconstruction.DirectReconstruction` estimates CSMs, DCFs,
# and performs an adjoint Fourier transform.
# This the simplest reconstruction method in our high-level interface to the reconstruction pipeline.

# %%
import mrpro
import torch

# %% [markdown]
# ### Load the data
# We load in the Data from the ISMRMRD file. We want use the trajectory that is stored also stored the ISMRMRD file.
# This can be done by passing a `~mrpro.data.traj_calculators.KTrajectoryIsmrmrd` object to
# `~mrpro.data.KData.from_file` when loading creating the `~mrpro.data.KData`.

# %%
trajectory_calculator = mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
kdata = mrpro.data.KData.from_file(data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5', trajectory_calculator)

# %% [markdown]
### Setup the DirectReconstruction instance
# We create a `~mrpro.algorithms.reconstruction.DirectReconstruction` and supply ``kdata``.
# `~mrpro.algorithms.reconstruction.DirectReconstruction` uses the information in ``kdata`` to
#  setup a Fourier transfrm, density compensation factors, and estimate coil sensitivity maps.
# (See the *Behind the scenes* section for more details.)
#
# ```{note}
# You can also directly set the Fourier operator, coil sensitivity maps, density compensation factors, etc.
# of the reconstruction instance.
# ```

# %%
reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)

# %% [markdown]
# All reconstruction algorithms in mrpro are implemented as PyTorch modules and can be moved to the GPU.
# In general, both the data and the reconstruction module must be moved to the same device.

# %%
if torch.cuda.is_available():
    # Move the data to the GPU if available
    reconstruction = reconstruction.cuda()
    kdata = kdata.cuda()

# %% [markdown]
### Perform the reconstruction
# The reconstruction is performed by calling the passing the k-space data.
# ```{note}
# Often, the data used to obtain the meta data for constructing the reconstruction instance
# is the same as the data passed to the reconstruction.
# But you can also different to create the coil sensitivity maps, dcf, etc.
# than the data that is passed to the reconstruction.
# ```

# %%
img = reconstruction(kdata)

# %% [markdown]
# ### Display the reconstructed image
# We now got in `mrpro.data.IData` object containing a header and the image tensor.
# We display the reconstructed image using matplotlib.

# %%
import matplotlib.pyplot as plt

# If there are multiple slices, ..., only the first one is selected
first_img = img.rss().cpu()[0, 0]  #  images, z, y, x
plt.imshow(first_img, cmap='gray')
plt.axis('off')
plt.show()

# %% [markdown]
# ## Behind the scenes
# These steps are done in a direct reconstruction:
#
# ### Calculate dcf using the trajectory
# The density compensation factors are calculated using the voronoi method.
# %%
dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()

# %% [markdown]
# ### Setup Fourier Operetor
# The Fourier operator is created using the trajectory and header information in kdata.

# %%
fourier_operator = dcf_operator @ mrpro.operators.FourierOp.from_kdata(kdata)
adjoint_operator = fourier_operator.H

# %% [markdown]
# ### Calculate coil sensitivity maps
# Coil sensitivity maps are calculated using the walsh method.

# %%
img_coilwise = mrpro.data.IData.from_tensor_and_kheader(*adjoint_operator(kdata.data), kdata.header)
csm_operator = mrpro.data.CsmData.from_idata_walsh(img_coilwise).as_operator()

# %% [markdown]
# ### Perform Direct Reconstruction
# Finally, the direct reconstruction is performed and an `mrpro.data.IData` object with the reconstructed
# image is returned.
# %%
adjoint_operator = (fourier_operator @ csm_operator).H

img_manual = mrpro.data.IData.from_tensor_and_kheader(*adjoint_operator(kdata.data), kdata.header)

# %% [markdown]
# ## Further behind the scenes
# ... these steps are equivalent to:

# %%
# Define Fourier operator manually
fourier_operator = mrpro.operators.FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)

# Calculate 2D dcf from the trajectory using the voronoi method
kykx = torch.stack((kdata.traj.ky[0, 0], kdata.traj.kx[0, 0]))
dcf_tensor = mrpro.algorithms.dcf.dcf_2d3d_voronoi(kykx)

# Perform density compensated adjoint Fourier transform
(img_tensor_coilwise,) = (fourier_operator.H * dcf_tensor)(kdata.data)

# Calculate and apply coil maps
csm_data = mrpro.algorithms.csm.walsh(img_tensor_coilwise[0], smoothing_width=5)
csm_operator = mrpro.operators.SensitivityOp(csm_data)
(img_tensor_coilcombined,) = csm_operator.adjoint(img_tensor_coilwise)
img_more_manual = mrpro.data.IData.from_tensor_and_kheader(img_tensor_coilcombined, kdata.header)
# %% [markdown]
# ### Check for equal results
# The 3 versions result should in the same image data.
# %%
# If the assert statement did not raise an exception, the results are equal.
assert torch.allclose(img.data, img_manual.data)
assert torch.allclose(img.data, img_more_manual.data)
