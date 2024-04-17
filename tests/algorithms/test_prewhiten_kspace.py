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

import pytest
import torch
from einops import rearrange
from mrpro.algorithms._prewhiten_kspace import prewhiten_kspace
from mrpro.data import KData
from mrpro.data import KNoise
from mrpro.data import KTrajectory
from tests import RandomGenerator


def _calc_coil_cov(data):
    data = rearrange(data, '... coils k2 k1 k0->coils (... k2 k1 k0)')
    cov = (1.0 / (data.shape[1])) * torch.einsum('ax,bx->ab', data, data.conj())
    return cov


def _test_prewhiten_kspace(random_kheader, device):
    """Prewhitening of k-space data."""

    # Dimensions
    n_coils = 4
    ns_k2k1k0 = (4, 5, 32)
    ns_other = (1,)

    # Create random noise samples
    random_data = RandomGenerator(0).complex64_tensor((*ns_other, n_coils, *ns_k2k1k0))
    knoise = KNoise(data=random_data).to(device=device)

    # Whiten KData created with **same** data and dummy trajectory
    trajectory = KTrajectory(
        torch.zeros(*ns_other, *ns_k2k1k0), torch.zeros(*ns_other, *ns_k2k1k0), torch.zeros(*ns_other, *ns_k2k1k0)
    )
    kdata = KData(header=random_kheader, data=random_data, traj=trajectory).to(device=device)
    kdata_white = prewhiten_kspace(kdata, knoise)

    # This should result in a covariance matrix that is the identity matrix
    expected_covariance = torch.eye(n_coils, dtype=torch.complex64, device=device)
    covariance = _calc_coil_cov(kdata_white.data)
    torch.testing.assert_close(covariance, expected_covariance)


def test_prewhiten_kspace_cpu(random_kheader):
    """Prewhitening of k-space data on CPU."""
    _test_prewhiten_kspace(random_kheader, 'cpu')


@pytest.mark.cuda()
def test_prewhiten_kspace_cuda(random_kheader):
    """Prewhitening of k-space data on GPU."""
    _test_prewhiten_kspace(random_kheader, 'cuda')
