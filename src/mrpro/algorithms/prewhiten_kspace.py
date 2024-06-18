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

from copy import deepcopy

import torch
from einops import einsum
from einops import parse_shape
from einops import rearrange

from mrpro.data._kdata.KData import KData
from mrpro.data.KNoise import KNoise


def prewhiten_kspace(kdata: KData, knoise: KNoise, scale_factor: float | torch.Tensor = 1.0) -> KData:
    """Calculate noise prewhitening matrix and decorrelate coils.

    This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

    Step 1: Calculate noise correlation matrix N
    Step 2: Carry out Cholesky decomposition L L^H = N
    Step 3: Estimate noise decorrelation matrix D = inv(L)
    Step 4: Apply D to k-space data

    More information can be found in
    http://onlinelibrary.wiley.com/doi/10.1002/jmri.24687/full
    https://doi.org/10.1002/mrm.1910160203

    If the the data has more samples in the 'other'-dimensions (batch/slice/...),
    the noise covariance matrix is calculated jointly over all samples.
    Thus, if the noise is not stationary, the noise covariance matrix is not accurate.
    In this case, the function should be called for each sample separately.

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
        Prewhitened copy of k-space data
    """
    # Reshape noise to (coil, everything else)
    noise = rearrange(knoise.data, '... coils k2 k1 k0->coils (... k2 k1 k0)')

    # Calculate noise covariance matrix and Cholesky decomposition
    noise_cov = (1.0 / (noise.shape[-1])) * einsum(
        noise, noise.conj(), 'coil1 everythingelse, coil2 everythingelse -> coil1 coil2'
    )

    cholsky = torch.linalg.cholesky(noise_cov)

    # solve_triangular is numerically more stable than inverting the matrix
    # but requires a single batch dimension
    kdata_flat = rearrange(kdata.data, '... coil k2 k1 k0 -> (... k2 k1 k0) coil 1')
    whitened_flat = torch.linalg.solve_triangular(cholsky, kdata_flat, upper=False)
    whitened_flatother = rearrange(
        whitened_flat, '(other k2 k1 k0) coil 1-> other coil k2 k1 k0', **parse_shape(kdata.data, '... k2 k1 k0')
    )
    whitened_data = whitened_flatother.reshape(kdata.data.shape) * torch.as_tensor(scale_factor).sqrt()
    header = deepcopy(kdata.header)
    traj = deepcopy(kdata.traj)

    return KData(header, whitened_data, traj)
