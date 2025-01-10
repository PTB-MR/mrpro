# %% [markdown]
# # Basics of MRpro and Cartesian Reconstructions
# Here, we are going to have a look at a few basics of MRpro and reconstruct data acquired with a Cartesian sampling
# pattern.
# %% [markdown]
# ## Overview
# In this notebook, we are going to explore the `mrpro.data.KData` object and the included header parameters.
# We will then use a FFT-operator in order to reconstruct data acquired with a Cartesian sampling scheme.
# We will also reconstruct data  acquired on a Cartesian grid but with partial echo and partial Fourier acceleration.
# Finally, we will reconstruct a Cartesian scan with regular undersampling using iterative SENSE.


# %% tags=["hide-cell"]
# Get the raw data from zenodo
import tempfile
from pathlib import Path

import zenodo_get

data_folder = Path(tempfile.mkdtemp())
dataset = '14173489'
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %% [markdown]
# We have three different scans obtained from the same object with the same FOV and resolution, saved as ISMRMRD
# raw data files (``*.mrd`` or ``*.h5``):
#
# - ``cart_t1.mrd`` is a fully sampled Cartesian acquisition
#
# - ``cart_t1_msense_integrated.mrd`` is accelerated using regular undersampling and self-calibrated SENSE
#
# - ``cart_t1_partial_echo_partial_fourier.mrd`` is accelerated using partial echo and partial Fourier


# %% [markdown]
# ## Read in raw data and explore header
#
# To read in an ISMRMRD file, we can simply pass on the file name to a `~mrpro.data.KData` object.
# Additionally, we need to provide information about the trajectory. In MRpro, this is done using trajectory
# calculators. These are functions that calculate the trajectory based on the acquisition information and additional
# parameters provided to the calculators (e.g. the angular step for a radial acquisition).
#
# In this case, we have a Cartesian acquisition. This means that we only need to provide a Cartesian trajectory
# calculator `~mrpro.data.traj_calculators.KTrajectoryCartesian` without any further parameters.

# %%
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryCartesian

kdata = KData.from_file(data_folder / 'cart_t1.mrd', KTrajectoryCartesian())

# %% [markdown]
# Now we can explore this data object.
# Simply printing ``kdata`` gives us a basic overview of the `~mrpro.data.KData` object.

# %% tags=["show-output"]
print(kdata)


# %% [markdown]
# We can also have a look at more specific header information like the 1H Lamor frequency

# %%
print('Lamor Frequency:', kdata.header.lamor_frequency_proton)

# %% [markdown]
# ## Reconstruction of fully sampled acquisition
#
# For the reconstruction of a fully sampled Cartesian acquisition, we can use a simple Fast Fourier Transform (FFT).
#
# Let's create an FFT-operator `mrpro.operator.FastFourierOp` and apply it to our `~mrpro.data.KData` object.
# Please note that all MRpro operator work on PyTorch tensors and not on the MRpro objects directly. Therefore, we have
# to call the operator on kdata.data. One other important property of MRpro operators is that they always return a
# tuple of PyTorch tensors, even if the output is only a single tensor. This is why we use the ``(img,)`` syntax below.

# %%
from mrpro.operators import FastFourierOp

fft_op = FastFourierOp(dim=(-2, -1))
(img,) = fft_op.adjoint(kdata.data)

# %% [markdown]
# Let's have a look at the shape of the obtained tensor.

# %%
print('Shape:', img.shape)

# %% [markdown]
# We can see that the second dimension, which is the coil dimension, is 16. This means we still have a coil resolved
# dataset (i.e. one image for each coil element). We can use a simply root-sum-of-squares approach to combine them into
# one. Later, we will do something a bit more sophisticated. We can also see that the x-dimension is 512. This is
# because in MRI we commonly oversample the readout direction by a factor 2 leading to a FOV twice as large as we
# actually need. We can either remove this oversampling along the readout direction or we can simply tell the
# `~mrpro.operatoers.FastFourierOp` to crop the image by providing the correct output matrix size ``recon_matrix``.

# %%
# Create FFT-operator with correct output matrix size
fft_op = FastFourierOp(
    dim=(-2, -1),
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
)

(img,) = fft_op.adjoint(kdata.data)
print('Shape:', img.shape)

# %% [markdown]
# Now, we have an image which is 256 x 256 voxel as we would expect. Let's combine the data from the different receiver
# coils using root-sum-of-squares and then display the image. Note that we usually index from behind in MRpro
# (i.e. -1 for the last, -4 for the fourth last (coil) dimension) to allow for more than one 'other' dimension.

# %% tags=["hide-cell"]
import matplotlib.pyplot as plt
import torch


# plot the image
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
# Combine data from different coils and show magntiude image
magnitude_fully_sampled = img.abs().square().sum(dim=-4).sqrt().squeeze()
show_images(magnitude_fully_sampled)
# %% [markdown]
# Great! That was very easy! Let's try to reconstruct the next dataset.

# %% [markdown]
# ## Reconstruction of acquisition with partial echo and partial Fourier

# %% tags=["remove-output"]
# Read in the data
kdata_pe_pf = KData.from_file(data_folder / 'cart_t1_partial_echo_partial_fourier.mrd', KTrajectoryCartesian())

# Create FFT-operator with correct output matrix size
fft_op = FastFourierOp(
    dim=(-2, -1),
    recon_matrix=kdata.header.recon_matrix,
    encoding_matrix=kdata.header.encoding_matrix,
)

# Reconstruct coil resolved image(s)
(img_pe_pf,) = fft_op.adjoint(kdata_pe_pf.data)

# Combine data from different coils using root-sum-of-squares
magnitude_pe_pf = img_pe_pf.abs().square().sum(dim=-4).sqrt().squeeze()


# Plot both images
show_images(magnitude_fully_sampled, magnitude_pe_pf, titles=['fully sampled', 'PF & PE'])
# %% [markdown]
# Well, we got an image, but when we compare it to the previous result, it seems like the head has shrunk.
# Since that's extremely unlikely, there's probably a mistake in our reconstruction.
#
# Let's step back and check out the trajectories for both scans.

# %%
print(kdata.traj)

# %% [markdown]
# We see that the trajectory has ``kz``, ``ky``, and ``kx`` components. ``kx`` and ``ky`` only vary along one dimension.
# This is because MRpro saves meta data such as trajectories in an efficient way, where dimensions in which the data
# does not change are often collapsed. The original shape can be obtained by
# [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
# Here, to get the full trajectory as a tensor, we can also just call `~mrpro.data.KTrajectory.as_tensor()`:

# %%
# Plot the fully sampled trajectory (in blue)
full_kz, full_ky, full_kx = kdata.traj.as_tensor()
plt.plot(full_ky[0, 0].flatten(), full_kx[0, 0].flatten(), 'ob')

# Plot the partial echo and partial Fourier trajectory (in red)
full_kz, full_ky, full_kx = kdata_pe_pf.traj.as_tensor()
plt.plot(full_ky[0, 0].flatten(), full_kx[0, 0].flatten(), '+r')

plt.show()
# %% [markdown]
# We see that for the fully sampled acquisition, the k-space is covered symmetrically from -256 to 255 along the
# readout direction and from -128 to 127 along the phase encoding direction. For the acquisition with partial Fourier
# and partial echo acceleration, this is of course not the case and the k-space is asymmetrical.
#
# Our FFT-operator does not know about this and simply assumes that the acquisition is symmetric and any difference
# between encoding and recon matrix needs to be zero-padded symmetrically.
#
# To take the asymmetric acquisition into account and sort the data correctly into a matrix where we can apply the
# FFT-operator to, we have got the `~mrpro.operators.CartesianSamplingOp` in MRpro. This operator performs
# sorting based on the k-space trajectory and the dimensions of the encoding k-space.
#
# Let's try it out!

# %%
from mrpro.operators import CartesianSamplingOp

cart_sampling_op = CartesianSamplingOp(encoding_matrix=kdata_pe_pf.header.encoding_matrix, traj=kdata_pe_pf.traj)

# %% [markdown]
# Now, we first apply the adjoint CartesianSamplingOp and then call the adjoint FFT-operator.

# %%
(img_pe_pf,) = fft_op.adjoint(cart_sampling_op.adjoint(kdata_pe_pf.data)[0])
magnitude_pe_pf = img_pe_pf.abs().square().sum(dim=-4).sqrt().squeeze()

show_images(magnitude_fully_sampled, magnitude_pe_pf, titles=['fully sampled', 'PF & PE'])

# %% [markdown]
# Voila! We've got the same brains, and they're the same size!

# %% [markdown]
# In MRpro, we have a smart `~mrpro.operators.FourierOp` operator, that automatically does the resorting and can
# handle non-cartesian data as well. For cartesian data, it internally does exactly the two steps we just did manually.
# The operator can be also be created from an existing `~mrpro.data.KData` object
# This is the recommended way to transform k-space data.

# %%
from mrpro.operators import FourierOp

fourier_op = FourierOp.from_kdata(kdata_pe_pf)
(img_pe_pf,) = fourier_op.adjoint(kdata_pe_pf.data)
magnitude_pe_pf = img_pe_pf.abs().square().sum(dim=-4).sqrt().squeeze()
show_images(magnitude_fully_sampled, magnitude_pe_pf, titles=['fully sampled', 'PF & PE'])

# %% [markdown]
# That was easy!
# But wait a second — something still looks a bit off. In the bottom left corner, it seems like there's a "hole"
# in the brain. That definitely shouldn't be there.
#
# The issue is that we combined the data from the different coils using a root-sum-of-squares approach.
# While it's simple, it's not the ideal method. Typically, coil sensitivity maps are calculated to combine the data
# from different coils. In MRpro, you can do this by calculating coil sensitivity data and then creating a
# `mrpro.operators.SensitivityOp` to combine the data after image reconstruction.


# %% [markdown]
# We have different options for calculating coil sensitivity maps from the image data of the various coils.
# Here, we're going to use the Walsh method.

# %%
from mrpro.algorithms.csm import walsh
from mrpro.operators import SensitivityOp

# Calculate coil sensitivity maps
(magnitude_pe_pf,) = fft_op.adjoint(cart_sampling_op.adjoint(kdata_pe_pf.data)[0])

# This algorithms is designed to calculate coil sensitivity maps for each other dimension.
csm_data = walsh(magnitude_pe_pf[0, ...], smoothing_width=5)[None, ...]

# Create SensitivityOp
csm_op = SensitivityOp(csm_data)

# Reconstruct coil-combined image
(img_pe_pf,) = csm_op.adjoint(*fourier_op.adjoint(img_pe_pf))
magnitude_pe_pf = magnitude_pe_pf.abs().squeeze()
show_images(magnitude_fully_sampled, magnitude_pe_pf, titles=['fully sampled', 'PF & PE'])


# %% [markdown]
# Tada! The "hole" is gone, and the image looks much better.
#
# When we reconstructed the image, we called the adjoint method of several different operators one after the other. That
# was a bit cumbersome. To make our life easier, MRpro allows to combine the operators first and then call the adjoint
# of the composite operator. We have to keep in mind that we have to put them in the order of the forward method of the
# operators. By calling the adjoint, the order will be automatically reversed.

# %%
# Create composite operator
adjoint_operator = (fourier_op @ csm_op).H
(magnitude_pe_pf,) = adjoint_operator(kdata_pe_pf.data)
magnitude_pe_pf = magnitude_pe_pf.abs().squeeze()
show_images(magnitude_pe_pf, titles=['PF & PE'])

# %% [markdown]
# Although we now have got a nice looking image, it was still a bit cumbersome to create it. We had to define several
# different operators and chain them together. Wouldn't it be nice if this could be done automatically?
#
# That is why we also included some top-level reconstruction algorithms in MRpro. For this whole steps from above,
# we can simply use a `mrpro.algorithnms.reconstruction.DirectReconstruction`.
# Reconstruction algorithms can be instantiated from only the information in the `~mrpro.data.KData` object.
#
# In contrast to operators, top-level reconstruction algorithms operate on the data objects of MRpro, i.e. the input is
# a `~mrpro.data.KData` object and the output is an `~mrpro.data.IData` object containing
# the reconstructed image data. To get its magnitude, we can call the `~mrpro.data.IData.rss` method.

# %%
from mrpro.algorithms.reconstruction import DirectReconstruction

# Create DirectReconstruction object from KData object
direct_recon_pe_pf = DirectReconstruction(kdata_pe_pf)

# Reconstruct image by calling the DirectReconstruction object
idat_pe_pf = direct_recon_pe_pf(kdata_pe_pf)


# %% [markdown]
# This is much simpler — everything happens in the background, so we don't have to worry about it.
# Let's finally try it on the undersampled dataset now.


# %% [markdown]
# ## Reconstruction of undersampled data

# %%
kdata_us = KData.from_file(data_folder / 'cart_t1_msense_integrated.mrd', KTrajectoryCartesian())
direct_recon_us = DirectReconstruction(kdata_us)
idat_us = direct_recon_us(kdata_us)

show_images(idat_pe_pf.rss().squeeze(), idat_us.rss().squeeze(), titles=['PE & PF', 'Undersampled'])

# %% [markdown]
# As expected, we can see undersampling artifacts in the image. In order to get rid of them, we can use an iterative
# SENSE algorithm. As you might have guessed, this is also included in MRpro.

# Similarly to the `~mrpro.algorithms.reconstruction.DirectReconstruction`,
# we can use an `~mrpro.algorithms.reconstruction.IterativeSENSEReconstruction`.
# For more information, see <project:iterative_sense_reconstruction>
