"""Tests for KTrajectoryRawShape class."""

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
import torch
from einops import rearrange
from einops import repeat
from mrpro.data import KTrajectoryRawShape


def make_trajectory_raw_shape(ksomething, sort_idx):
    """Reshape kx, ky or kz from (other k2 k1 k0) to ((other k2 k1) k0) and
    permute order with sort_idx."""
    # Reshape to raw trajectory shape
    ksomething = rearrange(ksomething, 'other k2 k1 k0->(other k2 k1) k0')

    # Permute trajectory points using sort_idx
    ksomething_perm = torch.zeros_like(ksomething)
    ksomething_perm[sort_idx, :] = ksomething
    return ksomething_perm


def test_trajectory_raw_reshape(cartesian_grid):
    """Test reshaping between raw KTrajectory and KTrajectory."""
    # Create Cartesian raw trajectory
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)

    # Repeat other dimension (e.g. multiple slices)
    n_other = 3
    kz_full = repeat(kz_full, '1 k2 k1 k0->other k2 k1 k0', other=n_other)
    ky_full = repeat(ky_full, '1 k2 k1 k0->other k2 k1 k0', other=n_other)
    kx_full = repeat(kx_full, '1 k2 k1 k0->other k2 k1 k0', other=n_other)

    # Reshape and repeat from (n_other n_k2 n_k1 n_k0) to (n_other*n_k2*n_k1 n_k0) and permute randomly
    sort_idx = np.random.default_rng(0).permutation(np.linspace(0, n_other * n_k2 * n_k1 - 1, n_other * n_k2 * n_k1))
    kz_raw = make_trajectory_raw_shape(kz_full, sort_idx)
    ky_raw = make_trajectory_raw_shape(ky_full, sort_idx)
    kx_raw = make_trajectory_raw_shape(kx_full, sort_idx)

    # Create raw trajectory
    trajectory_raw = KTrajectoryRawShape(kz_raw, ky_raw, kx_raw)

    # Reshape to original trajectory
    trajectory = trajectory_raw.reshape(sort_idx, n_k2, n_k1, repeat_detection_tolerance=None)

    # Compare trajectories
    torch.testing.assert_close(trajectory.kz, kz_full)
    torch.testing.assert_close(trajectory.ky, ky_full)
    torch.testing.assert_close(trajectory.kx, kx_full)
