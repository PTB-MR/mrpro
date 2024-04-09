"""Tests for k-space prewhitening function."""

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

import torch
from einops import rearrange
from mrpro.algorithms._prewhiten_kspace import prewhiten_kspace
from mrpro.data import KData
from mrpro.data import KNoise
from mrpro.data import KTrajectory


def _calc_coil_cov(data):
    data = rearrange(data, 'other coils k2 k1 k0->coils (other k2 k1 k0)')
    cov = (1.0 / (data.shape[1])) * torch.einsum('ax,bx->ab', data, data.conj())
    return cov


def test_prewhiten_kspace(random_kheader):
    """Prewhitening of k-space data."""

    # Dimensions
    n_coils = 4
    n_kx = 128

    # Create random noise samples
    knoise_data = torch.randn(1, n_coils, 1, 1, n_kx, dtype=torch.complex64)
    knoise = KNoise(data=knoise_data)

    # Create KData from same data with random trajectory
    trajectory = KTrajectory(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
    kdata = KData(header=random_kheader, data=knoise_data, traj=trajectory)

    kdata = prewhiten_kspace(kdata, knoise)
    torch.testing.assert_close(_calc_coil_cov(kdata.data), torch.eye(n_coils, dtype=torch.complex64))
