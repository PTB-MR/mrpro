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
from tests import RandomGenerator


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
