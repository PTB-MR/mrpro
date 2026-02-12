# %% [markdown]
# # Direct reconstruction of 2D golden angle radial data
# Here we use the `~mr2.algorithms.reconstruction.DirectReconstruction` class to perform a basic reconstruction of
# 2D radial data.
# A *direct* reconstruction uses the density compensated adjoint of the acquisition operator to obtain the images.

# %% [markdown]
# ## Using `~mr2.algorithms.reconstruction.DirectReconstruction`
# We use the `~mr2.algorithms.reconstruction.DirectReconstruction` class to reconstruct images from 2D radial data.
# `~mr2.algorithms.reconstruction.DirectReconstruction` estimates sensitivity maps, density compensation factors, etc.
# and performs an adjoint Fourier transform.
# This the simplest reconstruction method in our high-level interface to the reconstruction pipeline.

# %% [markdown]
# ### Load the data
# We load in the Data from the ISMRMRD file. We want use the trajectory that is stored also stored the ISMRMRD file.
# This can be done by passing a `~mr2.data.traj_calculators.KTrajectoryIsmrmrd` object to
# `~mr2.data.KData.from_file` when loading creating the `~mr2.data.KData`.

# %% tags=["hide-cell"]  mystnb={"code_prompt_show": "Show download details"}
# Download raw data from Zenodo
import os
import tempfile
from pathlib import Path

import zenodo_get

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.download(
    record='14617082', retry_attempts=5, output_dir=data_folder, access_token=os.environ.get('ZENODO_TOKEN')
)

# %%
import mr2
import torch

trajectory_calculator = mr2.data.traj_calculators.KTrajectoryIsmrmrd()
kdata = mr2.data.KData.from_file(data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5', trajectory_calculator)

# %% [markdown]
### Setup the DirectReconstruction instance
# We create a `~mr2.algorithms.reconstruction.DirectReconstruction` and supply ``kdata``.
# `~mr2.algorithms.reconstruction.DirectReconstruction` uses the information in ``kdata`` to
#  setup a Fourier transfrm, density compensation factors, and estimate coil sensitivity maps.
# (See the *Behind the scenes* section for more details.)
#
# ```{note}
# You can also directly set the Fourier operator, coil sensitivity maps, density compensation factors, etc.
# of the reconstruction instance.
# ```

# %%
reconstruction = mr2.algorithms.reconstruction.DirectReconstruction(kdata)

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
# We now got in `~mr2.data.IData` object containing a header and the image tensor.
# We display the reconstructed image using matplotlib.

# %%
import matplotlib.pyplot as plt

# If there are multiple slices, ..., only the first one is selected
first_img = img.rss()[0, 0]  #  images, z, y, x
plt.imshow(first_img, cmap='gray')
plt.axis('off')
plt.show()

# %% [markdown]
# ## Behind the scenes
# We now peek behind the scenes to see what happens in the `~mr2.algorithms.reconstruction.DirectReconstruction`
# class, and perform all steps manually:
# - Calculate density compensation factors
# - Setup Fourier operator
# - Obtain coil-wise images
# - Calculate coil sensitivity maps
# - Perform direct reconstruction

# ### Calculate density compensation using the trajectory
# We use a Voronoi tessellation of the trajectory to calculate the `~mr2.data.DcfData` and obtain
# a `~mr2.operators.DensityCompensationOp` operator.

# %%
dcf_operator = mr2.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()

# %% [markdown]
# ### Setup Fourier Operator
# Next, we create the Fourier operator. We can just pass the ``kdata`` object to the constructor of the
# `~mr2.operators.FourierOp`, and the trajectory and header information is used to create the operator. We want the
# to use the adjoint density compensated Fourier operator, so we perform a composition with ``dcf_operator``
# and use the `~mr2.operators.FourierOp.H` property of the operator to obtain its adjoint.

# %%
fourier_operator = dcf_operator @ mr2.operators.FourierOp.from_kdata(kdata)
adjoint_operator = fourier_operator.H

# %% [markdown]
# ### Calculate coil sensitivity maps
# Coil sensitivity maps are calculated using the walsh method (See `~mr2.data.CsmData` for other available methods).
# We first need to calculate the coil-wise images, which are then used to calculate the coil sensitivity maps.

# %%
img_coilwise = mr2.data.IData.from_tensor_and_kheader(*adjoint_operator(kdata.data), kdata.header)
csm_operator = mr2.data.CsmData.from_idata_walsh(img_coilwise).as_operator()

# %% [markdown]
# ### Perform Direct Reconstruction
# Finally, the direct reconstruction is performed and an `~mr2.data.IData` object with the reconstructed
# image is returned. We update the ``adjoint_operator`` to also include the coil sensitivity maps, thus
# performing the coil combination.

# %%
adjoint_operator = (fourier_operator @ csm_operator).H
img_manual = mr2.data.IData.from_tensor_and_kheader(*adjoint_operator(kdata.data), kdata.header)

# %% [markdown]
# ## Further behind the scenes
# There is also a even more manual way to perform the direct reconstruction. We can set up the Fourier operator by
# passing the trajectory and matrix sizes.

# %%
fourier_operator = mr2.operators.FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
# %% [markdown]
# We can call one of the algorithms in `mr2.algorithms.dcf` to calculate the density compensation factors.

# %%
kykx = torch.stack((kdata.traj.ky[0, 0], kdata.traj.kx[0, 0]))
dcf_tensor = mr2.algorithms.dcf.dcf_2d3d_voronoi(kykx)

# %% [markdown]
# We use these DCFs to weight the k-space data before performing the adjoint Fourier transform. We can also call
# `~mr2.operators.FourierOp.adjoint` on the Fourier operator instead of obtaining an adjoint operator.

# %%
(img_tensor_coilwise,) = fourier_operator.adjoint(dcf_tensor * kdata.data)

# %% [markdown]
# Next, we calculate the coil sensitivity maps by using one of the algorithms in `mr2.algorithms.csm` and set
# up a `~mr2.operators.SensitivityOp` operator.

# %%
csm_data = mr2.algorithms.csm.walsh(img_tensor_coilwise[0], smoothing_width=5)
csm_operator = mr2.operators.SensitivityOp(csm_data)

# %% [markdown]
# Finally, we perform the coil combination of the coil-wise images and obtain final images.

# %%
(img_tensor_coilcombined,) = csm_operator.adjoint(img_tensor_coilwise)
img_more_manual = mr2.data.IData.from_tensor_and_kheader(img_tensor_coilcombined, kdata.header)

# %% [markdown]
# ### Check for equal results
# The 3 versions result should in the same image data.

# %%
# If the assert statement did not raise an exception, the results are equal.
torch.testing.assert_close(img.data, img_manual.data)
torch.testing.assert_close(img.data, img_more_manual.data, atol=1e-4, rtol=1e-4)
