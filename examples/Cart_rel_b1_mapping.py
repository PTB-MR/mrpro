# %% [markdown]
# # Script for B1+ mapping from H5 file.
# This example script reconstructs a Cartesian dataset.

# %% import functionality
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def b1reco(
    idata: torch.Tensor, relphasechannel: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a relative B1 map."""
    # relphasechannel ... use channel XXX as reference
    # idata ... TX+2 GRE data

    # shift the input data to have [X, Y, Z, RX, MEAS]
    ima = torch.moveaxis(idata, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    sz = ima.shape

    # calculate the noise level
    noise_scan = ima[:, :, :, :, 1]
    noise_mean = torch.mean(torch.abs(torch.flatten(noise_scan))) / 1.253  # factor due to rician distribution
    rx_sens = (
        torch.mean(torch.mean(torch.mean(torch.abs(noise_scan), 0), 0), 0)
    ) / 1.253  # factor due to rician distribution

    # calculate the noise correlation
    nn = torch.reshape(noise_scan, (sz[0] * sz[1] * sz[2], sz[3]))
    noise_corr = torch.complex(torch.zeros(sz[3], sz[3]), torch.zeros(sz[3], sz[3]))

    for il in range(sz[3]):
        for im in range(sz[3]):
            nnsubset = torch.cat((nn[:, il, None], nn[:, im, None]), dim=1)
            nnsubset = torch.moveaxis(nnsubset, [0, 1], [1, 0])
            cc = torch.corrcoef(nnsubset)
            noise_corr[il, im] = cc[1, 0]

    # correct for different RX sensitivities
    ima_cor = ima[:, :, :, :, 2:] / rx_sens

    # calculate the relative TX phase
    phasetemp = ima_cor / ima_cor[:, :, :, :, relphasechannel, None]
    phasetemp[~torch.isfinite(phasetemp.abs())] = 0.0

    cxtemp = torch.sum(torch.abs(ima_cor) * torch.exp(1j * torch.angle(phasetemp)), dim=3, keepdim=True)
    cxtemp2 = torch.moveaxis(cxtemp, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])
    b1p_phase = torch.exp(1j * torch.angle(cxtemp2[:, :, :, :]))

    # calculate the TX magnitude as in PFVM ISMRM abstract
    imamag = torch.abs(ima_cor)
    b1_magtmp = torch.sum(imamag, dim=3, keepdim=True) / (
        (torch.sum(torch.sum(imamag, dim=3, keepdim=True), dim=4, keepdim=True)) ** 0.5
    )
    b1p_mag = torch.moveaxis(b1_magtmp, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])
    sum_cp = torch.sqrt(torch.sum(torch.abs(torch.sum(ima_cor, dim=4, keepdim=True)) ** 2, dim=3, keepdim=True))

    rk = torch.sum(imamag, dim=3, keepdim=True) / sum_cp
    rk = torch.squeeze(torch.moveaxis(rk, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3]))

    # calculate the relative RX phase and magnitude
    ima_cor_tmp = ima_cor[:, :, :, relphasechannel, :]
    ima_cor_tmp = ima_cor_tmp[:, :, :, None, :]

    phasetemp = ima_cor / ima_cor_tmp
    phasetemp[~torch.isfinite(phasetemp.abs())] = 0.0

    cxtemp = torch.sum(torch.abs(ima_cor) * torch.exp(1j * torch.angle(phasetemp)), dim=4, keepdim=True)

    b1m_phase = torch.exp(1j * torch.angle(cxtemp[:, :, :, :]))
    b1m_mag = torch.sum(imamag, dim=4, keepdim=True) / (
        (torch.sum(torch.sum(imamag, dim=3, keepdim=True), dim=4, keepdim=True)) * 0.5
    )

    b1p_mag = torch.moveaxis(b1p_mag, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    b1m_mag = torch.moveaxis(b1m_mag, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])

    b1p_phase = torch.moveaxis(b1p_phase, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    b1m_phase = torch.moveaxis(b1m_phase, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])

    return b1m_mag, b1m_phase, b1p_mag, b1p_phase, noise_mean


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
