""" Fast B1 Mapping using NTx GRE """
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


def B1reco(IData, relphasechannel):
    # relphasechannel ... use channel XXX as reference
    # IData ... TX+2 GRE data

    import torch

    # shift the input data to have [X, Y, Z, RX, MEAS]
    ima = torch.moveaxis(IData, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    sz = ima.shape

    # calculate the noise level
    noise_scan = ima[:, :, :, :, 1]
    noise_mean = torch.mean(torch.abs(torch.flatten(noise_scan))) / 1.253  # factor due to rician distribution
    RX_sens = (
        torch.mean(torch.mean(torch.mean(torch.abs(noise_scan), 0), 0), 0)
    ) / 1.253  # factor due to rician distribution

    # calculate the noise correlation
    nn = torch.reshape(noise_scan, (sz[0] * sz[1] * sz[2], sz[3]))
    noise_corr = torch.complex(torch.zeros(sz[3], sz[3]), torch.zeros(sz[3], sz[3]))

    for lL in range(0, sz[3]):
        for lM in range(0, sz[3]):
            nnsubset = torch.cat((nn[:, lL, None], nn[:, lM, None]), dim=1)
            nnsubset = torch.moveaxis(nnsubset, [0, 1], [1, 0])
            cc = torch.corrcoef(nnsubset)
            noise_corr[lL, lM] = cc[1, 0]

    # correct for different RX sensitivities
    ima_cor = ima[:, :, :, :, 2:] / RX_sens

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
    sum_cp = torch.sqrt(
        torch.sum(torch.abs(torch.sum(ima_cor, dim=4, keepdim=True)) ** 2, dim=3, keepdim=True)
    )
    # sum_cp = torch.squeeze(sum_cp)

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
