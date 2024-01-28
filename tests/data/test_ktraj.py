"""Tests for KTrajectory class."""

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

import numpy as np
import pytest
import torch

from mrpro.data import KTrajectory
from mrpro.data.enums import TrajType
from tests import RandomGenerator
from tests.conftest import COMMON_MR_TRAJECTORIES


@pytest.fixture(params=({'seed': 0},))
def cartesian_grid(request):
    generator = RandomGenerator(request.param['seed'])

    def generate(nk2: int, nk1: int, nk0: int, jitter: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k0_range = torch.arange(nk0)
        k1_range = torch.arange(nk1)
        k2_range = torch.arange(nk2)
        ky, kz, kx = torch.meshgrid(
            k1_range,
            k2_range,
            k0_range,
            indexing='xy',
        )
        if jitter > 0:
            kx = kx + generator.float32_tensor((nk2, nk1, nk0), high=jitter)
            ky = ky + generator.float32_tensor((nk2, nk1, nk0), high=jitter)
            kz = kz + generator.float32_tensor((nk2, nk1, nk0), high=jitter)
        return kz.unsqueeze(0), ky.unsqueeze(0), kx.unsqueeze(0)

    return generate


def create_uniform_traj(nk, k_shape):
    """Create a tensor of uniform points with predefined shape nk."""
    kidx = torch.where(torch.tensor(nk[1:]) > 1)[0]
    if len(kidx) > 1:
        raise ValueError('nk is allowed to have at most one non-singleton dimension')
    if len(kidx) >= 1:
        # kidx+1 because we searched in nk[1:]
        n_kpoints = nk[kidx + 1]
        # kidx+2 because k_shape also includes coils dimensions
        k = torch.linspace(-k_shape[kidx + 2] // 2, k_shape[kidx + 2] // 2 - 1, n_kpoints, dtype=torch.float32)
        views = [1 if i != n_kpoints else -1 for i in nk]
        k = k.view(*views).expand(list(nk))
    else:
        k = torch.zeros(nk)
    return k


def create_traj(k_shape, nkx, nky, nkz, sx, sy, sz):
    """Create trajectory with random entries."""
    random_generator = RandomGenerator(seed=0)
    k_list = []
    for spacing, nk in zip([sz, sy, sx], [nkz, nky, nkx]):
        if spacing == 'nuf':
            k = random_generator.float32_tensor(size=nk)
        elif spacing == 'uf':
            k = create_uniform_traj(nk, k_shape=k_shape)
        elif spacing == 'z':
            k = torch.zeros(nk)
        k_list.append(k)
    ktraj = KTrajectory(*k_list)
    return ktraj


def test_ktraj_repeat_detection_tol(cartesian_grid):
    """Test the automatic detection of repeated values."""
    nk0 = 10
    nk1 = 20
    nk2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(nk2, nk1, nk0, jitter=0.1)
    ktraj_sparse = KTrajectory(kz_full, ky_full, kx_full, repeat_detection_tolerance=0.1)

    assert ktraj_sparse.broadcasted_shape == (1, nk2, nk1, nk0)
    assert ktraj_sparse.kx.shape == (1, 1, 1, nk0)
    assert ktraj_sparse.ky.shape == (1, 1, nk1, 1)
    assert ktraj_sparse.kz.shape == (1, nk2, 1, 1)


def test_ktraj_repeat_detection_exact(cartesian_grid):
    """Test the automatic detection of repeated values."""
    nk0 = 10
    nk1 = 20
    nk2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(nk2, nk1, nk0, jitter=0.1)
    ktraj_full = KTrajectory(kz_full, ky_full, kx_full, repeat_detection_tolerance=None)

    assert ktraj_full.broadcasted_shape == (1, nk2, nk1, nk0)
    assert ktraj_full.kx.shape == (1, nk2, nk1, nk0)
    assert ktraj_full.ky.shape == (1, nk2, nk1, nk0)
    assert ktraj_full.kz.shape == (1, nk2, nk1, nk0)


def test_ktraj_tensor_conversion(cartesian_grid):
    nk0 = 10
    nk1 = 20
    nk2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(nk2, nk1, nk0, jitter=0.0)
    ktraj = KTrajectory(kz_full, ky_full, kx_full)
    tensor = torch.stack((kz_full, ky_full, kx_full), dim=0)

    tensor_from_traj = ktraj.as_tensor()  # stack_dim=0
    tensor_from_traj_dim2 = ktraj.as_tensor(stack_dim=2).moveaxis(2, 0)
    tensor_from_traj_from_tensor_dim3 = KTrajectory.from_tensor(tensor.moveaxis(0, 3), stack_dim=3).as_tensor()
    tensor_from_traj_from_tensor = KTrajectory.from_tensor(tensor).as_tensor()  # stack_dim=0

    torch.testing.assert_close(tensor, tensor_from_traj)
    torch.testing.assert_close(tensor, tensor_from_traj_dim2)
    torch.testing.assert_close(tensor, tensor_from_traj_from_tensor)
    torch.testing.assert_close(tensor, tensor_from_traj_from_tensor_dim3)


def test_ktraj_raise_not_broadcastable():
    """Non broadcastable shapes should raise."""
    kx = ky = torch.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    kz = torch.arange(1 * 2 * 3 * 100).reshape(1, 2, 3, 100)
    with pytest.raises(ValueError):
        ktraj = KTrajectory(kz, ky, kx)


def test_ktraj_raise_wrong_dim():
    """Wrong number of dimensions after broadcasting should raise."""
    kx = ky = kz = torch.arange(1 * 2 * 3).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        ktraj = KTrajectory(kz, ky, kx)


def test_ktraj_to_float64(cartesian_grid):
    """Change KTrajectory dtype to float64."""
    nk0 = 10
    nk1 = 20
    nk2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(nk2, nk1, nk0, jitter=0.0)
    ktraj = KTrajectory(kz_full, ky_full, kx_full)

    ktraj_float64 = ktraj.to(dtype=torch.float64)
    assert ktraj_float64.kz.dtype == torch.float64
    assert ktraj_float64.ky.dtype == torch.float64
    assert ktraj_float64.kx.dtype == torch.float64


@pytest.mark.cuda
def test_ktraj_cuda(cartesian_grid):
    """Move KTrajectory object to CUDA memory."""
    nk0 = 10
    nk1 = 20
    nk2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(nk2, nk1, nk0, jitter=0.0)
    ktraj = KTrajectory(kz_full, ky_full, kx_full)

    ktraj_cuda = ktraj.cuda()
    assert ktraj_cuda.kz.is_cuda
    assert ktraj_cuda.ky.is_cuda
    assert ktraj_cuda.kx.is_cuda


@pytest.mark.cuda
def test_ktraj_cpu(cartesian_grid):
    """Move KTrajectory object to CUDA memory and back to CPU memory."""
    nk0 = 10
    nk1 = 20
    nk2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(nk2, nk1, nk0, jitter=0.0)
    ktraj = KTrajectory(kz_full, ky_full, kx_full)

    ktraj_cpu = ktraj.cuda().cpu()
    assert ktraj_cpu.kz.is_cpu
    assert ktraj_cpu.ky.is_cpu
    assert ktraj_cpu.kx.is_cpu


def test_traj_type_update():
    """Test update of trajectory type when trajectory changes."""
    # Generate random RPE trajectory
    k_shape = (1, 8, 8, 64, 96)
    nkx = (1, 1, 1, 96)
    nky = (1, 8, 64, 1)
    nkz = (1, 8, 64, 1)
    sx = 'uf'  # uniform sampling along readout
    sy = 'nuf'
    sz = 'nuf'
    ktraj = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)

    # Make readout non-uniform
    ktraj.kx[:] = RandomGenerator(seed=0).float32_tensor(size=nkx)

    with pytest.raises(NotImplementedError, match='modified'):
        ktraj.type_along_k210

    with pytest.raises(NotImplementedError, match='modified'):
        ktraj.type_along_kzyx


@COMMON_MR_TRAJECTORIES
def test_ktype_along_kzyx(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz, s0, s1, s2):
    """Test identification of traj types."""

    # Generate random k-space trajectories
    ktraj = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)

    # Find out the type of the kz, ky and kz dimensions
    single_value_dims = [d for d, s in zip((-3, -2, -1), (sz, sy, sx)) if s == 'z']
    on_grid_dims = [d for d, s in zip((-3, -2, -1), (sz, sy, sx)) if s == 'uf']
    not_on_grid_dims = [d for d, s in zip((-3, -2, -1), (sz, sy, sx)) if s == 'nuf']

    # check dimensions which are of shape 1 and do not need any transform
    assert all([ktraj.type_along_kzyx[dim] & TrajType.SINGLEVALUE for dim in single_value_dims])

    # Check dimensions which are on a grid and require FFT
    assert all([ktraj.type_along_kzyx[dim] & TrajType.ONGRID for dim in on_grid_dims])

    # Check dimensions which are not on a grid and require NUFFT
    assert all([~(ktraj.type_along_kzyx[dim] & (TrajType.SINGLEVALUE | TrajType.ONGRID)) for dim in not_on_grid_dims])


@COMMON_MR_TRAJECTORIES
def test_ktype_along_k210(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz, s0, s1, s2):
    """Test identification of traj types."""

    # Generate random k-space trajectories
    ktraj = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)

    # Find out the type of the k2, k1 and k0 dimensions
    single_value_dims = [d for d, s in zip((-3, -2, -1), (s2, s1, s0)) if s == 'z']
    on_grid_dims = [d for d, s in zip((-3, -2, -1), (s2, s1, s0)) if s == 'uf']
    not_on_grid_dims = [d for d, s in zip((-3, -2, -1), (s2, s1, s0)) if s == 'nuf']

    # check dimensions which are of shape 1 and do not need any transform
    assert all([ktraj.type_along_k210[dim] & TrajType.SINGLEVALUE for dim in single_value_dims])

    # Check dimensions which are on a grid and require FFT
    assert all([ktraj.type_along_k210[dim] & TrajType.ONGRID for dim in on_grid_dims])

    # Check dimensions which are not on a grid and require NUFFT
    assert all([~(ktraj.type_along_k210[dim] & (TrajType.SINGLEVALUE | TrajType.ONGRID)) for dim in not_on_grid_dims])
