"""Script for B1+ mapping from H5 file."""
# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Christoph Aigner, 2024.03.22
#
# RPE example with 3D volume works
# example data can be downloaded from:
# https://figshare.com/articles/dataset/High-Resolution_3D_Radial_Phase_Encoded_GRE_of_8_Transmit_Channels_with_a_Siemens_7T_pTx_System_Phantom_VB17_RAW_Data_/24316519

# %% import functionality
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators._KTrajectorySunflowerGoldenRpe import KTrajectorySunflowerGoldenRpe
from mrpro.operators import FourierOp

# Download the file from `url` and save it locally under `file_name`:
with (
    urllib.request.urlopen('https://figshare.com/ndownloader/files/43259838') as response,
    open('meas_MID335_B1R_FA_20_cv_pTX_sun_B1R_v1p1_FID39870_ismrmrd.h5', 'wb') as out_file,  # noqa PTH123
):
    data = response.read()  # a `bytes` object
    out_file.write(data)


from B1reco import b1reco

# %% B1+ mapping - RPE 3D
#
# define the file name of the 3D RPE raw data
h5_filename = R'meas_MID335_B1R_FA_20_cv_pTX_sun_B1R_v1p1_FID39870_ismrmrd.h5'

# Load the channelwise GRE data for relative B1+ mapping
data = KData.from_file(
    ktrajectory=KTrajectorySunflowerGoldenRpe(),
    filename=h5_filename,
)

# manually set up the recon and encoding matrix
data.header.recon_matrix.x = 128
data.header.recon_matrix.y = 80
data.header.recon_matrix.z = 80

data.header.encoding_matrix.x = 128
data.header.encoding_matrix.y = 16
data.header.encoding_matrix.z = 16

# perform FT, shift the k-space center and create IData object
op = FourierOp(recon_matrix=data.header.recon_matrix, encoding_matrix=data.header.encoding_matrix, traj=data.traj)

# set up the density compensation
dc = DcfData.from_traj_voronoi(data.traj)

# reconstruct the density compensated data
(images,) = op.H(data.data * dc.data)

# create IData object from image tensor and kheader
idata = IData.from_tensor_and_kheader(images, data.header)

# run B1reco
b1m_mag, b1m_pha, b1p_mag, b1p_pha, noise_mean = b1reco(idata.data, relphasechannel=0)

# plot results
fig, axis = plt.subplots(2, 4, figsize=(16, 8))
for i, axs in enumerate(axis.flatten()):
    axs.imshow(b1p_mag[0, i, :, :, 64])

fig, axis = plt.subplots(2, 4, figsize=(16, 8))
for i, axs in enumerate(axis.flatten()):
    axs.imshow(np.angle(b1p_pha[0, i, :, :, 64]), vmin=-3.15, vmax=3.15)

# %%
