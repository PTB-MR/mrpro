"""Script for B1+ mapping from MRD file."""
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
#   Christoph Aigner, 2023.03.22
#
# Cartesian example with 2D slice works
# example data can be downloaded from:
# https://figshare.com/articles/dataset/High-Resolution_3D_Radial_Phase_Encoded_GRE_of_8_Transmit_Channels_with_a_Siemens_7T_pTx_System_Phantom_VB17_RAW_Data_/24316519

# %% import functionality
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

# Download the file from `url` and save it locally under `file_name`:
with (
    urllib.request.urlopen('https://figshare.com/ndownloader/files/43259835') as response,
    open('meas_MID296_ssm_CVB1R_1sl_sag_trig400_FID39837_ismrmrd.h5', 'wb') as out_file,  # noqa PTH123
):
    data = response.read()  # a `bytes` object
    out_file.write(data)

from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators._KTrajectoryCartesian import KTrajectoryCartesian
from mrpro.operators import FourierOp

from B1reco import b1reco

# %% Cartesian B1+ mapping - 1 2D Slice
# Load the channelwise GRE data for relative B1+ mapping

h5_filename = R'meas_MID296_ssm_CVB1R_1sl_sag_trig400_FID39837_ismrmrd.h5'

data = KData.from_file(
    ktrajectory=KTrajectoryCartesian(),
    filename=h5_filename,
)

# perform FT and shift the k-space center
op = FourierOp(recon_matrix=data.header.recon_matrix, encoding_matrix=data.header.encoding_matrix, traj=data.traj)

# reconstruct the density compensated data
(images,) = op.H(data.data)

# create IData object from image tensor and kheader
idata = IData.from_tensor_and_kheader(images, data.header)

b1m_mag, b1m_pha, b1p_mag, b1p_pha, noise_mean = b1reco(idata.data, relphasechannel=0)

# plot results
fig, axis = plt.subplots(2, 4, figsize=(16, 8))
for i, axs in enumerate(axis.flatten()):
    axs.imshow(b1p_mag[0, i, 0, :, :])

fig, axis = plt.subplots(2, 4, figsize=(16, 8))
for i, axs in enumerate(axis.flatten()):
    axs.imshow(np.angle(b1p_pha[0, i, 0, :, :]))

# %%
