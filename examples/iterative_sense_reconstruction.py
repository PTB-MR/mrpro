# %% [markdown]
# # Direct Reconstruction of 2D golden angle radial data
# Here we use the DirectReconstruction class to reconstruct images from ISMRMRD 2D radial data
# %%
# define zenodo URL of the example ismrmd data
zenodo_url = 'https://zenodo.org/records/10854057/files/'
fname = 'pulseq_radial_2D_402spokes_golden_angle_with_traj.h5'
# %%
# Download raw data
from pathlib import Path

# TODO: replace with automatic download
data_folder = Path('/Users/kolbit01/Documents/PTB/Data/mrpro/raw/')

# %% [markdown]
# ### Image reconstruction
# We use the DirectReconstruction class to reconstruct images from 2D radial data.
# DirectReconstruction estimates CSMs, DCFs and performs an adjoint Fourier transform.
# This is a high-level interface to the reconstruction pipeline.
# %%
import mrpro

# Use the trajectory that is stored in the ISMRMRD file
trajectory = mrpro.data.traj_calculators.KTrajectoryIsmrmrd()
# Load in the Data from the ISMRMRD file
kdata = mrpro.data.KData.from_file(data_folder / '2D_GRad_map_t1_traj_2s.h5', trajectory)

# Perform direct reconstruction
reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction.from_kdata(kdata)
# Use this to run on gpu: kdata = kdata.cuda()
img = reconstruction(kdata)

# %%
import matplotlib.pyplot as plt

# Display the reconstructed image
# If there are multiple slices, ..., only the first one is selected
first_img = img.rss().cpu()[0, 0, :, :]  #  images, z, y, x
plt.matshow(first_img, cmap='gray')

# %%


# %% [markdown]
# Copyright 2024 Physikalisch-Technische Bundesanstalt
# Apache License 2.0. See LICENSE file for details.
