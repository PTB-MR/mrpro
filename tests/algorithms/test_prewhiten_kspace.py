"""Tests for k-space prewhitening function."""

import pytest
import torch
from einops import rearrange
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.data import KData, KNoise, KTrajectory
from mrpro.utils import RandomGenerator


def _calc_coil_cov(data):
    data = rearrange(data, '... coils k2 k1 k0 -> coils (... k2 k1 k0)')
    cov = (1.0 / data.shape[1]) * torch.einsum('ax,bx->ab', data, data.conj())
    return cov


@pytest.mark.parametrize('kdata_tensor', [True, False], ids=['kdata_tensor', 'kdata_object'])
@pytest.mark.parametrize('knoise_tensor', [True, False], ids=['knoise_tensor', 'knoise_object'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_prewhiten_kspace(random_kheader, device, kdata_tensor, knoise_tensor):
    """Test prewhitening for tensor and KData/KNoise inputs."""

    n_coils = 4
    n_k2k1k0 = (4, 5, 32)
    n_other = (2, 3)

    random_data = RandomGenerator(0).complex64_tensor((*n_other, n_coils, *n_k2k1k0)).to(device=device)

    knoise: KNoise | torch.Tensor = random_data if knoise_tensor else KNoise(data=random_data)

    kdata: KData | torch.Tensor
    if kdata_tensor:
        kdata = random_data
    else:
        trajectory = KTrajectory(
            torch.zeros(*n_other, *n_k2k1k0), torch.zeros(*n_other, *n_k2k1k0), torch.zeros(*n_other, *n_k2k1k0)
        )
        kdata = KData(header=random_kheader, data=random_data, traj=trajectory).to(device=device)

    kdata_white = prewhiten_kspace(kdata, knoise)

    assert isinstance(kdata_white, torch.Tensor if kdata_tensor else KData)

    whitened = kdata_white.data if isinstance(kdata_white, KData) else kdata_white

    expected_covariance = torch.eye(n_coils, dtype=torch.complex64, device=device)
    covariance = _calc_coil_cov(whitened)

    torch.testing.assert_close(covariance, expected_covariance)
