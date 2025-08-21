# %% [markdown]
# # Direct reconstruction of 2D golden angle radial data
# Here we use the `~mrpro.algorithms.reconstruction.DirectReconstruction` class to perform a basic reconstruction of
# 2D radial data.
# A *direct* reconstruction uses the density compensated adjoint of the acquisition operator to obtain the images.

# %% [markdown]
# ## Using `~mrpro.algorithms.reconstruction.DirectReconstruction`
# We use the `~mrpro.algorithms.reconstruction.DirectReconstruction` class to reconstruct images from 2D radial data.
# `~mrpro.algorithms.reconstruction.DirectReconstruction` estimates sensitivity maps, density compensation factors, etc.
# and performs an adjoint Fourier transform.
# This the simplest reconstruction method in our high-level interface to the reconstruction pipeline.

# %% [markdown]
# ### Load the data
# We load in the Data from the ISMRMRD file. We want use the trajectory that is stored also stored the ISMRMRD file.
# This can be done by passing a `~mrpro.data.traj_calculators.KTrajectoryIsmrmrd` object to
# `~mrpro.data.KData.from_file` when loading creating the `~mrpro.data.KData`.

# %% tags=["hide-cell"]  mystnb={"code_prompt_show": "Show download details"}
# Download raw data from Zenodo
import tempfile
from pathlib import Path

import zenodo_get

tmp = tempfile.TemporaryDirectory()  # RAII, automatically cleaned up
data_folder = Path(tmp.name)
zenodo_get.download(record='14617082', retry_attempts=5, output_dir=data_folder)

# %%
import mrpro
import torch

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
# We now got in `~mrpro.data.IData` object containing a header and the image tensor.
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
# We now peek behind the scenes to see what happens in the `~mrpro.algorithms.reconstruction.DirectReconstruction`
# class, and perform all steps manually:
# - Calculate density compensation factors
# - Setup Fourier operator
# - Obtain coil-wise images
# - Calculate coil sensitivity maps
# - Perform direct reconstruction

# ### Calculate density compensation using the trajectory
# We use a Voronoi tessellation of the trajectory to calculate the `~mrpro.data.DcfData` and obtain
# a `~mrpro.operators.DensityCompensationOp` operator.

# %%
dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()

# %% [markdown]
# ### Setup Fourier Operator
# Next, we create the Fourier operator. We can just pass the ``kdata`` object to the constructor of the
# `~mrpro.operators.FourierOp`, and the trajectory and header information is used to create the operator. We want the
# to use the adjoint density compensated Fourier operator, so we perform a composition with ``dcf_operator``
# and use the `~mrpro.operators.FourierOp.H` property of the operator to obtain its adjoint.

# %%
fourier_operator = dcf_operator @ mrpro.operators.FourierOp.from_kdata(kdata)
adjoint_operator = fourier_operator.H

# %% [markdown]
# ### Calculate coil sensitivity maps
# Coil sensitivity maps are calculated using the walsh method (See `~mrpro.data.CsmData` for other available methods).
# We first need to calculate the coil-wise images, which are then used to calculate the coil sensitivity maps.

# %%
img_coilwise = mrpro.data.IData.from_tensor_and_kheader(*adjoint_operator(kdata.data), kdata.header)
csm_operator = mrpro.data.CsmData.from_idata_walsh(img_coilwise).as_operator()

# %% [markdown]
# ### Perform Direct Reconstruction
# Finally, the direct reconstruction is performed and an `~mrpro.data.IData` object with the reconstructed
# image is returned. We update the ``adjoint_operator`` to also include the coil sensitivity maps, thus
# performing the coil combination.

# %%
adjoint_operator = (fourier_operator @ csm_operator).H
img_manual = mrpro.data.IData.from_tensor_and_kheader(*adjoint_operator(kdata.data), kdata.header)

# %% [markdown]
# ## Further behind the scenes
# There is also a even more manual way to perform the direct reconstruction. We can set up the Fourier operator by
# passing the trajectory and matrix sizes.

# %%
fourier_operator = mrpro.operators.FourierOp(
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
    traj=kdata.traj,
)
# %% [markdown]
# We can call one of the algorithms in `mrpro.algorithms.dcf` to calculate the density compensation factors.

# %%
kykx = torch.stack((kdata.traj.ky[0, 0], kdata.traj.kx[0, 0]))
dcf_tensor = mrpro.algorithms.dcf.dcf_2d3d_voronoi(kykx)

# %% [markdown]
# We use these DCFs to weight the k-space data before performing the adjoint Fourier transform. We can also call
# `~mrpro.operators.FourierOp.adjoint` on the Fourier operator instead of obtaining an adjoint operator.

# %%
(img_tensor_coilwise,) = fourier_operator.adjoint(dcf_tensor * kdata.data)

# %% [markdown]
# Next, we calculate the coil sensitivity maps by using one of the algorithms in `mrpro.algorithms.csm` and set
# up a `~mrpro.operators.SensitivityOp` operator.

# %%
csm_data = mrpro.algorithms.csm.walsh(img_tensor_coilwise[0], smoothing_width=5)
csm_operator = mrpro.operators.SensitivityOp(csm_data)

# %% [markdown]
# Finally, we perform the coil combination of the coil-wise images and obtain final images.

# %%
(img_tensor_coilcombined,) = csm_operator.adjoint(img_tensor_coilwise)
img_more_manual = mrpro.data.IData.from_tensor_and_kheader(img_tensor_coilcombined, kdata.header)

# %% [markdown]
# ### Check for equal results
# The 3 versions result should in the same image data.

# %%
# If the assert statement did not raise an exception, the results are equal.
torch.testing.assert_close(img.data, img_manual.data)
torch.testing.assert_close(img.data, img_more_manual.data, atol=1e-4, rtol=1e-4)


# %% [markdown]
# Faster DCF
# %%
from mrpro.data import KTrajectory
from mrpro.operators import FourierOp


def estimate_dcf(trajectory: KTrajectory, fourier_op: FourierOp, max_iter: int = 0) -> torch.Tensor:
    """Estimate the density compensation factors for a given trajectory.

    Uses the Jackson or Pipe method to estimate the density of an arbitrary set of points.
    If max_iter is set to 0, the Jackson method is used. Otherwise, the Pipe method is used.

    Parameters
    ----------
    trajectory
        Shap  `(*other, 2 or 3, k2, k1, k0)`
    fourier_op
        The Fourier operator
    max_iter
        The number of iterations to use for the Pipe method. If set to 0, the Jackson method is used.

    Returns
    -------
    density estimate

    References
    ----------
      .. [1] Jackson, J.I., Meyer, C.H., Nishimura, D.G. and Macovski, A. (1991),
        Selection of a convolution function for Fourier inversion using gridding
        (computerized tomography application). IEEE Transactions on Medical
        Imaging, 10(3): 473-478. https://doi.org/10.1109/42.97598
      .. [2] Pipe, J.G. and Menon, P. (1999), Sampling density compensation in
        MRI: Rationale and an iterative numerical solution. Magn. Reson. Med.,
        41: 179-186. https://doi.org/10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V
    """
    ones = torch.ones(trajectory.broadcasted_shape, dtype=torch.complex64).unsqueeze(-4)
    if fourier_op._non_uniform_fast_fourier_op is None:
        # cartesian
        return ones
    op = fourier_op._non_uniform_fast_fourier_op @ fourier_op._non_uniform_fast_fourier_op.H
    weight = op(ones)[0].reciprocal().nan_to_num()
    for _ in range(max_iter):
        weight *= op(weight)[0].reciprocal().nan_to_num()
    return weight.abs().squeeze(-4)


# %%
fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata)
dcf_operator2 = mrpro.data.DcfData(estimate_dcf(kdata.traj, fourier_operator)).as_operator()
adjoint1 = (dcf_operator @ fourier_operator @ csm_operator).H
adjoint2 = (dcf_operator2 @ fourier_operator @ csm_operator).H
(img1,) = adjoint1(kdata.data)
(img2,) = adjoint2(kdata.data)

plt.matshow(img1.abs()[0, 0, 0], cmap='gray')
plt.title('DCFs from Voronoi')
plt.colorbar()
plt.matshow(img2.abs()[0, 0, 0], cmap='gray')
plt.title('DCFs from estimate_dcf')
plt.colorbar()
# %%
