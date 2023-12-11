""""Script for B1+ mapping from MRD file."""
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
# mypy: ignore-errors
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
from einops import rearrange
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectorySunflowerGoldenRpe
from mrpro.operators import FourierOp
from mrpro.data import SpatialDimension

# %% Define data path
# folder = R'Z:\_allgemein\projects\8_13\B1Mapping'
# file = R'meas_MID14_B1R_FA_20_cv_pTX_sun_B1R_v1p1_FID4365.h5'
# mrd_path = Path(folder) / file
# assert mrd_path.is_file()
# %% Create KData object
from mrpro.data._KData import KData
from mrpro.data.traj_calculators._KTrajectoryCartesian import KTrajectoryCartesian

# %%
h5_filename = (
    R'C:\Users\hammac01\Desktop\PythonCode\mrpro_test_data\meas_MID296_ssm_CVB1R_1sl_sag_trig400_FID39837_ismrmrd.h5'
)
data = KData.from_file(
    ktrajectory=KTrajectoryCartesian(),
    filename=h5_filename,
)
# %%
op = FourierOp(im_shape=SpatialDimension(1, 256, 512), traj=data.traj, oversampling=SpatialDimension(1, 1, 1))
im = torch.fft.fftshift(op.H(data.data))
image = torch.fft.fftshift(im.abs().square().sum(1).sqrt())
# sortidx = torch.argsort(data.traj.ky, dim=-2, stable=True)
# reshaped = torch.broadcast_to(sortidx.unsqueeze(1), data.data.shape)
# sorted = torch.gather(data.data, -2, reshaped)
# coilwise = torch.fft.fftshift(torch.fft.ifft2(sorted), dim=(-1, -2))
# image = coilwise.abs().square().sum(1).sqrt()
# im = image.squeeze()
# im = im[:, None, None, :, :]  # .squeeze()
# %% create IData object from image tensor and kheader
idata = IData.from_tensor_and_kheader(im, data.header)
# %% plot example img
plt.matshow(np.abs(idata.data[0, 0, 0, :, :]))
plt.show()
# %% CODE FROM MANUEL
# routes for B1 Mapping
opts = {}
opts['DIMB1'] = 2
opts['WHICHSLICES'] = 1
opts['FIGWIDTH'] = 16
opts['RELPHASECHANNEL'] = 1
opts['USEMEAN'] = True
opts['SHOWMAPS'] = False
opts['B1PSCALING'] = 1


def removeInf(inputMatrix):
    outputMatrix = inputMatrix
    outputMatrix = np.where(outputMatrix == np.inf, 0, outputMatrix)
    return outputMatrix


def removeNaN(inputMatrix):
    outputMatrix = inputMatrix
    outputMatrix = np.nan_to_num(outputMatrix, 0.0)
    return outputMatrix


def repmat(input_matrix, reps):
    len_reps = len(reps)
    len_input = len(input_matrix.shape)
    if len_input < len_reps:
        input_matrix_append = input_matrix
        while len(input_matrix_append.shape) < len_reps:
            input_matrix_append = np.reshape(input_matrix, (input_matrix_append.shape + (1,)))
        output_matrix = np.tile(input_matrix_append, reps)
    elif len_input == len_reps:
        output_matrix = np.tile(input_matrix, reps)
    elif len_input > len_reps:
        reps_append = reps
        while len(reps_append) < len_input:
            reps_append = reps_append + (1,)
        output_matrix = np.tile(input_matrix, reps_append)
    return output_matrix


def B1reco(IData):
    ima = np.moveaxis(IData, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    sz = ima.shape
    # calculate the noise level
    noise_scan = ima[:, :, :, :, 1]
    # noise_mean = np.mean(np.abs(noise_scan)) / 1.253
    # RX_sens = (
    #     np.mean(
    #         np.mean(np.mean(np.abs(noise_scan), axis=0, keepdims=True), axis=1, keepdims=True), axis=2, keepdims=True
    #     )
    #     / 1.253
    # )
    # calculate the correlation
    # coil correlation, can be resued from csm walshh?
    nn = np.reshape(noise_scan, (sz[0] * sz[1] * sz[2], sz[3]), order='F')
    noise_corr = np.zeros((sz[3], sz[3]), dtype=complex)
    for lL in range(0, sz[3]):
        for lM in range(0, sz[3]):
            cc = np.corrcoef(nn[:, lL], nn[:, lM])
            noise_corr[lL, lM] = cc[1, 0]
    # correct for different RX sensitivities
    ima_cor = ima[:, :, :, :, 2:]
    sz_cor = ima_cor.shape
    # calculate the relative TX phase
    # ima_cor_ref = np.sum(ima_cor, axis=4) / sz_cor[4]
    phasetemp = removeInf(
        removeNaN(np.divide(ima_cor, repmat(ima_cor[:, :, :, :, opts['RELPHASECHANNEL'] - 1], (1, 1, 1, 1, sz_cor[4]))))
    )
    cxtemp = np.sum(np.abs(ima_cor) * np.exp(1j * np.angle(phasetemp)), axis=3, keepdims=True)
    cxtemp2 = np.moveaxis(cxtemp, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])
    if cxtemp2.shape[2] != 1:
        cxtemp2 = np.squeeze(cxtemp2)
    else:
        cxtemp2 = np.squeeze(cxtemp2, axis=-1)
    # b1p_phase = np.exp(1j * np.angle(cxtemp2[:, :, :, :]))
    # calculate the TX magnitude
    if opts['USEMEAN']:
        # calculate as in ISMRM abstract
        imamag = np.abs(ima_cor)
        b1_magtmp = np.divide(
            np.sum(imamag, axis=3, keepdims=True),
            repmat(
                np.sum(np.sum(imamag, axis=3, keepdims=True), axis=4, keepdims=True) ** 0.5, (1, 1, 1, 1, sz_cor[4])
            ),
        )
        b1p_mag = np.moveaxis(b1_magtmp, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])
        if b1p_mag.shape[2] != 1:
            b1p_mag = np.squeeze(b1p_mag)
        else:
            b1p_mag = np.squeeze(b1p_mag, axis=-1)
        # calculate b1p_mag normalized with the CP mode
        sum_cp = np.sqrt(np.sum(np.abs(np.sum(ima_cor, axis=4, keepdims=True)) ** 2, axis=3, keepdims=True))
        sum_cp = np.squeeze(sum_cp)
        rk = np.divide(np.sum(imamag, axis=3, keepdims=True), repmat(sum_cp, (1, 1, 1, 1, sz_cor[4])))
        rk = np.moveaxis(rk, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])
        rk = np.squeeze(rk)
    else:
        # alternative calculation using median value
        imamag = np.abs(ima_cor)
        # b1_1 = np.divide(ima_cor, repmat(np.sum(np.abs(imamag), axis=4, keepdims=True), (1, 1, 1, 1, sz_cor[4])))
        # b1_2 = np.moveaxis(np.median(np.abs(b1_1), axis=3), [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])
        # b1p_magmed = np.multiply(
        #     b1_2,
        #     repmat(
        #         np.squeeze(np.sum(np.sum(np.abs(imamag), axis=4, keepdims=True), axis=3, keepdims=True) ** 0.5),
        #         (1, 1, 1, sz_cor[4]),
        #     ),
        # )
    # calculate the relative RX phase
    ima_cor_tmp = ima_cor[:, :, :, opts['RELPHASECHANNEL'] - 1, :]
    ima_cor_tmp = ima_cor_tmp[:, :, :, np.newaxis, :]
    phasetemp = removeInf(removeNaN(np.divide(ima_cor, repmat(ima_cor_tmp, (1, 1, 1, sz[3], 1)))))
    cxtemp = np.sum(np.abs(ima_cor) * np.exp(1j * np.angle(phasetemp)), axis=4, keepdims=True)
    if cxtemp.shape[2] != 1:
        cxtemp = np.squeeze(cxtemp)
    else:
        cxtemp = np.squeeze(cxtemp, axis=-1)
    # b1m_phase = np.exp(1j * np.angle(cxtemp[:, :, :, :]))
    # calculate TX magnitude
    b1m_mag = np.divide(
        np.sum(imamag, axis=4, keepdims=True),
        repmat(np.sum(np.sum(imamag, axis=3, keepdims=True), axis=4, keepdims=True) ** 0.5, (1, 1, 1, sz[3], 1)),
    )
    # if b1m_mag.shape[2] != 1:
    #     b1m_mag = np.squeeze(b1m_mag)
    # else:
    #     b1m_mag = np.squeeze(b1m_mag, axis=-1)
    # b1p_mag = np.moveaxis(b1p_mag, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    # b1m_mag = np.moveaxis(b1m_mag, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    """
    RPEB1p=permute(MR.Pars.Recon.B1p(end:-1:1,end:-1:1,:,:,:),[3 2 1 4 5]);%
    RPEB1m=permute(MR.Pars.Recon.B1m(end:-1:1,end:-1:1,:,:,:),[3 2 1 4 5]);%
    CVb1.b1p_mag = double( abs(RPEB1p));
    CVb1.b1p_pha = double(angle(RPEB1p));
    CVb1.b1m_mag = double( abs(RPEB1m));
    CVb1.b1m_pha = double(angle(RPEB1m));
    CVb1.orig_cxima = double( permute(DATA{1}(end:-1:1,end:-1:1,:,:,:),[3 2 1 4 5]));
    CVb1.b1pcx =double( RPEB1p);
    CVb1.b1mcx =double( RPEB1m);
    CVb1.noise_mean = MR.Pars.Recon.Noise_mean;
    CVb1.filename =   MR.Pars.Recon.SaveFile;
    B1p.cxmap = double( RPEB1p);
    """
    return b1p_mag, b1m_mag


# %% run B1reco
b1p_mag, b1m_mag = B1reco(idata.data.numpy())
# %%
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i, axs in enumerate(axs.flatten()):
    axs.imshow(b1p_mag[:, :, 0, i])
# %%
