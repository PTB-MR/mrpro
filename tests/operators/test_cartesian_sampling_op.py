"""Tests for the Cartesian sampling operator."""

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

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.operators import CartesianSamplingOp
from tests import RandomGenerator
from tests.data.test_ktraj import create_traj


def test_cart_sampling_op_data_match():
    # Create 3D uniform trajectory
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 60)
    nky = (1, 1, 40, 1)
    nkz = (1, 20, 1, 1)
    sx = 'uf'
    sy = 'uf'
    sz = 'uf'
    ktraj = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)

    # Create matching data
    random_generator = RandomGenerator(seed=0)
    kdata = random_generator.complex64_tensor(size=k_shape)

    # Create sampling operator
    encoding_shape = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    SOp = CartesianSamplingOp(encoding_shape=encoding_shape, traj=ktraj)

    # Subsample data and trajectory
    kdata_sub = kdata[:, :, ::2, ::4, ::3]
    ktraj_sub = KTrajectory(kz=ktraj.kz[:, ::2, :, :], ky=ktraj.ky[:, :, ::4, :], kx=ktraj.kx[:, :, :, ::3])
    SOp_sub = CartesianSamplingOp(encoding_shape=encoding_shape, traj=ktraj_sub)

    # Verify that the fully-sampled sampling operator does not do anything because the data is already sorted
    assert SOp._fft_idx is None

    # Verify identical shape
    (k,) = SOp.adjoint(kdata)
    (k_sub,) = SOp_sub.adjoint(kdata_sub)
    assert k.shape == k_sub.shape

    # Verify data is correctly sorted
    torch.testing.assert_close(kdata[:, :, ::2, ::4, ::3], k_sub[:, :, ::2, ::4, ::3])


@pytest.mark.parametrize(
    'sampling', ['random', 'partial_echo', 'partial_fourier', 'regular_undersampling', 'random_undersampling']
)
def test_cart_sampling_op_fwd_adj(sampling):
    """Test adjoint property of Cartesian sampling operator."""

    # Create 3D uniform trajectory
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 60)
    nky = (1, 1, 40, 1)
    nkz = (1, 20, 1, 1)
    sx = 'uf'
    sy = 'uf'
    sz = 'uf'
    ktraj = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)

    # Subsample data and trajectory
    if sampling == 'random':
        random_idx = torch.randperm(k_shape[-2])
        ktraj_sub = KTrajectory.from_tensor(ktraj.as_tensor()[..., random_idx, :])
    elif sampling == 'partial_echo':
        ktraj_sub = KTrajectory.from_tensor(ktraj.as_tensor()[..., : k_shape[-1] // 2])
    elif sampling == 'partial_fourier':
        ktraj_sub = KTrajectory.from_tensor(ktraj.as_tensor()[..., : k_shape[-3] // 2, : k_shape[-2] // 2, :])
    elif sampling == 'regular_undersampling':
        ktraj_sub = KTrajectory.from_tensor(ktraj.as_tensor()[..., ::3, ::5, :])
    elif sampling == 'random_undersampling':
        random_idx = torch.randperm(k_shape[-2])
        ktraj_sub = KTrajectory.from_tensor(ktraj.as_tensor()[..., random_idx[: k_shape[-2] // 2], :])
    else:
        raise ValueError(f'Test {sampling} not implemented.')

    encoding_shape = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    SOp_sub = CartesianSamplingOp(encoding_shape=encoding_shape, traj=ktraj_sub)

    # Test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=k_shape)
    v = random_generator.complex64_tensor(size=k_shape[:2] + ktraj_sub.as_tensor().shape[2:])
    (Fu,) = SOp_sub(u)
    (FHv,) = SOp_sub.H(v)
    Fu_v = torch.vdot(Fu.flatten(), v.flatten())
    u_FHv = torch.vdot(u.flatten(), FHv.flatten())

    # Check the adjoint property
    assert torch.isclose(Fu_v, u_FHv, rtol=1e-3)
