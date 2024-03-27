"""Prewhiten k-space data."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange

from mrpro.data._kdata._KData import KData
from mrpro.data._KNoise import KNoise


def prewhiten_kspace(kdata: KData, knoise: KNoise, scale_factor: float = 1.0) -> KData:
    """Calculate noise prewhitening matrix and decorrelate coils.

    This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

    Step 1: Calculate noise correlation matrix N
    Step 2: Carry out Cholesky decomposition L L^H = N
    Step 3: Estimate noise decorrelation matrix D = inv(L)
    Step 4: Apply D to k-space data

    More information can be found in
    http://onlinelibrary.wiley.com/doi/10.1002/jmri.24687/full
    https://doi.org/10.1002/mrm.1910160203

    Parameters
    ----------
    kdata
        K-space data.
    knoise
        Noise measurements.
    scale_factor
        Square root is applied on the noise covariance matrix. Used to adjust for effective noise bandwidth
        and difference in sampling rate between noise calibration and actual measurement:
        scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio, by default 1.0

    Returns
    -------
        Prewhitened k-space data
    """
    # Reshape noise to (coil, everything else)
    noise = rearrange(knoise.data, 'other coils k2 k1 k0->coils (other k2 k1 k0)')

    # Calculate noise covariance matrix which should ideally be a unity matrix, i.e. no noise correlation between coils
    noise_cov = (1.0 / (noise.shape[1])) * torch.einsum('ax,bx->ab', noise, noise.conj())

    # Calculate prewhitening matrix and scale
    prew = torch.linalg.inv(torch.linalg.cholesky(noise_cov)) * np.sqrt(scale_factor)

    # Apply prewhitening matrix
    prew_data = torch.einsum('yx,axbcd->aybcd', prew, kdata.data)

    return KData(kdata.header, prew_data, kdata.traj)
