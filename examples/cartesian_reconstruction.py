# %% [markdown]
# # Basics and Cartesian Reconstructions
# Here we are going to have a look at a few basics of MRpro and reconstruct data acquired with a Cartesian sampling
# pattern.

# %% [markdown]
# ## Overview
#
# In this notebook we are going to explore the MRpro KData object and the included header parameters. We will then use
# a FFT-operator in order to reconstruct data acquired with a Cartesian sampling scheme. We will also reconstruct data
# acquired on a Cartesian grid but with partial echo and partial Fourier acceleration. Finally we will reconstruct a
# Cartesian scan with regular undersampling using iterative SENSE.

# %% [markdown]
# ## Import MRpro and download data

# %%
# %%
# Get the raw data from zenodo
import tempfile
from pathlib import Path

import mrpro
import zenodo_get

data_folder = Path(tempfile.mkdtemp())
dataset = '14173489'
zenodo_get.zenodo_get([dataset, '-r', 5, '-o', data_folder])  # r: retries

# %%
# List the downloaded files
from os import listdir

for f in listdir(data_folder):
    print(f)

# %% [markdown]
# So we have three different scans obtained from the same object with the same FOV and resolution:
#
# - cart_t1.mrd is a fully sampled Cartesian acquisition
#
# - cart_t1_partial_echo_partial_fourier.mrd is accelerated using partial echo and partial Fourier
#
# - cart_t1_msense_integrated.mrd is accelerated using regular undersampling and self-calibrated SENSE

# %% [markdown]
# ## Read in raw data and explore header
#
# To read in a ISMRMRD raw data file we can simply pass on the file name to a KData object. In addition we also have to
# provide information about the trajectory. In MRpro this is done using trajectory calculators, which are functions that
# calculate the trajectory based on the acquisition information and additional parameters provided to the calculators
# (e.g. the angular step for a radial acquisition).
#
# In this case we have a Cartesian acquisition so we only need to provide a Cartesian trajectory calculator without any
# further parameters.

# %%
kdata = mrpro.data.KData.from_file(
    data_folder / Path('cart_t1.mrd'), mrpro.data.traj_calculators.KTrajectoryCartesian()
)

# %% [markdown]
# Now we can explore this data object.

# %%
# Start with simply calling print(kdata)
print(kdata)

# %%
# We can also have a look at the content of the header
print(kdata.header.acq_info.position[0, 0, 0, 0])

# %% [markdown]
# ## Reconstruction of fully sampled acquisition
#
# We have got a fully sampled Cartesian acquisition so we know we can use a Fast Fourier Transform (FFT) to
# reconstruction the data.
#
# Let's create an FFT-operator and apply it to kdata. Here it is important to note that all MRpro operators work on
# PyTorch tensors and not on the MRpro objects directly. Therefore we have to call the operator on kdata.data. One other
# important feature of MRpro operators is that the always return a tuple of length 1 of PyTorch tensors.

# %%
fft_op = mrpro.operators.FastFourierOp(dim=(-2, -1))
img = fft_op.adjoint(kdata.data)[0]

# %% [markdown]
# Let's have a look at the shape of the obtained tensor.

# %%
print(img.shape)

# %% [markdown]
# We can see that the second dimension which is the coil dimension is 16, so we still have a coil resolved dataset. We
# can use a simply root-sum-of-squares approach to combine them into one. Later we will do something a bit more
# sophisticated. We can also see that the x-dimension is 512. This is because in MRI we commonly oversample the readout
# direction by a factor 2 leading to a FOV twice as large as we actually need. We can either remove this oversampling
# along the readout direction or we can simply tell the FFT-operator to remove it by providing the correct output matrix
# size (recon_matrix).

# %%
fft_op = mrpro.operators.FastFourierOp(
    dim=(-2, -1), recon_matrix=kdata.header.recon_matrix, encoding_matrix=kdata.header.encoding_matrix
)
img = fft_op.adjoint(kdata.data)[0]
print(img.shape)

# %% [markdown]
# Now we have an image which is 256 x 256 voxel as we would expect. Let's combine the data from the different receiver
# coils using root-sum-of-squares and then display the image.

# %%
import torch

img = torch.sqrt(torch.sum(img**2, dim=1)).abs()

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img[0, 0])


# %% [markdown]
# ## Reconstruction of acquisition with partial echo and partial Fourier
#
# Great! That was very easy so let's try to reconstruct the next dataset.

# %%
kdata_pe_pf = mrpro.data.KData.from_file(
    data_folder / Path('cart_t1_partial_echo_partial_fourier.mrd'), mrpro.data.traj_calculators.KTrajectoryCartesian()
)
fft_op = mrpro.operators.FastFourierOp(
    dim=(-2, -1), recon_matrix=kdata.header.recon_matrix, encoding_matrix=kdata.header.encoding_matrix
)
img_pe_pf = fft_op.adjoint(kdata_pe_pf.data)[0]
img_pe_pf = torch.sqrt(torch.sum(img_pe_pf**2, dim=1)).abs()

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(img_pe_pf[0, 0])


# %% [markdown]
# Well we get an image out but when we compare it to the previous result it seems the head as shrunk. As this is not
# very likely there is probably a mistake in our reconstruction.
#
# Alright let's take a step back and have a look at the trajectory of both scans.

# %%
print(kdata.traj)

# %% [markdown]
# We can see that the trajectory consists of a kz, ky and kx part. Kx and ky only vary along a single dimension. The
# reason for this is that we try to save the trajectory in the most efficient way in MRpro. If we want to get the full
# trajectory as a tensor we can call as_tensor().

# %%
plt.figure()
plt.plot(kdata.traj.as_tensor()[2, 0, 0, :, :].flatten(), kdata.traj.as_tensor()[1, 0, 0, :, :].flatten(), 'ob')
plt.plot(
    kdata_pe_pf.traj.as_tensor()[2, 0, 0, :, :].flatten(), kdata_pe_pf.traj.as_tensor()[1, 0, 0, :, :].flatten(), '+r'
)

# %% [markdown]
# We can see that for the fully sampled acquisition the k-space is covered symmetrically from -256 to 255 along the
# readout direction and from -128 to 127 along the phase encoding direction. For the acquisition with partial Fourier
# and partial echo acceleration this is of course not the case and the k-space is asymmetrical.
#
# Our FFT-operator does not know about this and simply assumes that the acquisition is symmetric and any difference
# between encoding and recon matrix need to be zero-padded symmetrically.
#
# To take the asymmetric acquisition into account and sort the data correctly into a matrix where we can apply the
# FFT-operator to, we have got the CartesianSamplingOp in MRpro. This operator calculates a sorting index based on the
# k-space trajectory and the dimensions of the encoding k-space.

# %%
cart_sampling_op = mrpro.operators.CartesianSamplingOp(kdata_pe_pf.header.encoding_matrix, kdata_pe_pf.traj)

# %% [markdown]
# Now we can first apply the CartesianSamplingOp and then call the FFT-operator.

# %%
img_pe_pf = fft_op.adjoint(cart_sampling_op.adjoint(kdata_pe_pf.data)[0])[0]
img_pe_pf = torch.sqrt(torch.sum(img_pe_pf**2, dim=1)).abs()

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(img_pe_pf[0, 0])

# %% [markdown]
# Voila! Now we get the same brains with the same size. But hang on a second - there is still something which looks a
# bit funny. In the bottom left hand corner it seems that there is a "hole" in the brain. This should probably not be
# there.
#
# The reason for this is, that we simply combined the data from the different coils using a root-sum-of-squares
# approach. This was easy but not what we should do. Commonly coil sensitivity maps are calculated and they are then
# used to combine the data from the different coils. In MRpro this is done by calculating coil sensitivity data and
# then creating a SensitivityOp to combine the data after image reconstruction.

# %% [markdown]
# We have different option of how to calculate coil sensitivity maps from image data of the different coils. Here we
# are going to use the Walsh-method.

# %%
# Calculate coil sensitivity maps
img_pe_pf = fft_op.adjoint(cart_sampling_op.adjoint(kdata_pe_pf.data)[0])[0]
# This algorithms is designed to calculate coil sensitivity maps for each other dimension.
csm_data = mrpro.algorithms.csm.walsh(img_pe_pf[0, ...], smoothing_width=5)[None, ...]

# Create SensitivityOp
csm_op = mrpro.operators.SensitivityOp(csm_data)

# Reconstruct coil-combined image
img_pe_pf = csm_op.adjoint(fft_op.adjoint(cart_sampling_op.adjoint(kdata_pe_pf.data)[0])[0])[0].abs()

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(img_pe_pf[0, 0, 0])

# %% [markdown]
# Now we got an image without any "holes"!
#
# When we reconstructed the image we called the adjoint method of several different operators one after the other. That
# was a bit cumbersome. To make our life easier we can combine the operators directly and then call the adjoint of the
# composite operator. We have to keep in mind that we have to put them in the order of the forward method of the
# operators. By calling the adjoint, the order will be automatically reversed.

# %%
# Create composite operator
acq_op = cart_sampling_op @ fft_op @ csm_op
img_pe_pf = acq_op.adjoint(kdata_pe_pf.data)[0].abs()

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(img_pe_pf[0, 0, 0])

# %% [markdown]
# Although we have got now a nice looking image, it was a bit cumbersome to create it. We had to define several
# different operators and chain them together. Would we nice if this could be done automatically....
#
# That is why we also included some top-level reconstruction algorithms. The reconstruction above we simply call
# DirectReconstruction. A DirectReconstruction object can be directly created from the information in the KData object.
#
# Reconstruction algorithms operator on the data objects of MRpro, i.e. the input is a KData object and the output is a
# IData object. To visualize this we need the tensor content of the IData object which can be obtained by calling
# .rss().

# %%
direct_recon_pe_pf = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_pe_pf)
idat_pe_pf = direct_recon_pe_pf(kdata_pe_pf)

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(idat_pe_pf.rss()[0, 0])

# %% [markdown]
# This is much simpler and all of the magic is done in the background and we don't have to worry about it. Let's try it
# on the undersampled dataset.

# %% [markdown]
# ## Reconstruction of undersampled data

# %%
kdata_us = mrpro.data.KData.from_file(
    data_folder / Path('cart_t1_msense_integrated.mrd'), mrpro.data.traj_calculators.KTrajectoryCartesian()
)
direct_recon_us = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_us)
idat_us = direct_recon_us(kdata_us)

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(idat_us.rss()[0, 0])

# %% [markdown]
# As expected we can see undersampling artifacts in the image. In order to get rid of them we can use an iterative
# SENSE algorithm. Of course for this regularly undersampled we could also use Cartesian SENSE unfolding but in MRpro
# we don't have that.
#
# Similarly to the DirectReconstruction we can create an IterativeSENSEReconstruction and apply it to the undersampled
# KData. Give it a try and see if you can remove the undersampling artifacts.
#
# One important thing to keep in mind is, that this only works if the coil maps which we use, do not have any
# undersampling artifacts. Commonly we would get them from a fully sampled self-calibration reference lines in the
# center of k-space or a separate coil sensitivity scan.
#
# As a first step we are going to assume that we have got a nice fully sampled reference scan like our partial echo and
# partial Fourier acquisition. You can get the CsmData which is needed for the IterativeSENSEReconstruction by calling
# e.g. direct_recon_pe_pf.csm.

# %%
kdata_us = mrpro.data.KData.from_file(
    data_folder / Path('cart_t1_msense_integrated.mrd'), mrpro.data.traj_calculators.KTrajectoryCartesian()
)
it_sense_recon = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(kdata_us, csm=direct_recon_pe_pf.csm)
idat_us = it_sense_recon(kdata_us)

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(idat_us.rss()[0, 0])

# %% [markdown]
# That worked nicely but of course in practice we don't want to have to acquire a fully sampled version of our scan in
# order to be able to reconstruct our scan. A more efficient option is to obtain a few self-calibration lines in the
# center of k-space to make up a fully sampled low-resolution image.
#
# In our scan these lines are part of the dataset but they are not used for image reconstruction because they are
# labeled solely for calibration (i.e. calculation of coil sensitivity maps). Because they are not labeled for imaging,
# they are by default ignored by MRpro when we read in the data. We can set a flag when we call from_file in order to
# read in only those lines to reconstruct our coil sensitivity maps.

# %%
from mrpro.data.acq_filters import is_coil_calibration_acquisition

kdata_calib_lines = mrpro.data.KData.from_file(
    data_folder / Path('cart_t1_msense_integrated.mrd'),
    mrpro.data.traj_calculators.KTrajectoryCartesian(),
    acquisition_filter_criterion=lambda acq: is_coil_calibration_acquisition(acq),
)

direct_recon_calib_lines = mrpro.algorithms.reconstruction.DirectReconstruction(kdata_calib_lines)
im_calib_lines = direct_recon_calib_lines(kdata_calib_lines)

plt.imshow(im_calib_lines.rss()[0, 0, ...])

# %% [markdown]
# Although this only leads a low-resolution image, it is good enough to calculate coil sensitivity maps.

# %%
# Visualize coil sensitivity maps of all 16 coils
assert direct_recon_calib_lines.csm is not None
fig, ax = plt.subplots(4, 4, squeeze=False)
for idx, cax in enumerate(ax.flatten()):
    cax.imshow(direct_recon_calib_lines.csm.data[0, idx, 0, ...].abs())

# %% [markdown]
# Now we can use these coil sensitivity maps to reconstruct our SENSE scan.

# %%
kdata_us = mrpro.data.KData.from_file(
    data_folder / Path('cart_t1_msense_integrated.mrd'), mrpro.data.traj_calculators.KTrajectoryCartesian()
)
it_sense_recon = mrpro.algorithms.reconstruction.IterativeSENSEReconstruction(
    kdata_us, csm=direct_recon_calib_lines.csm
)
idat_us = it_sense_recon(kdata_us)

fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0, 0].imshow(img[0, 0])
ax[0, 1].imshow(idat_us.rss()[0, 0])

# %% [markdown]
# The final image is a little bit worse (nothing beats fully sampled high-resolution scans for coil map calculation)
# but we are able to get rid of the undersampling artifacts inside the brain. If you want to further improve the coil
# sensitivity map quality try:
# - use different methods to calculate them, e.g. mrpro.algorithms.csm.inati
# - play around with the parameters of these methods
# - apply a smoothing filter on the images (or ideally directly in k-space) used to calculate the coil sensitivity maps
