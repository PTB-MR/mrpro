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
import pytest
import torch
from einops import rearrange
from einops import repeat

from mrpro.data import KTrajectoryRawShape
from tests.data.test_ktraj import cartesian_grid


def make_ktraj_raw_shape(ksomething, sort_idx):
    """Reshape kx, ky or kz from (other k2 k1 k0) to ((other k2 k1) k0) and
    permute order with sort_idx."""
    # Reshape to raw trajectory shape
    ksomething = rearrange(ksomething, 'other k2 k1 k0->(other k2 k1) k0')

    # Permute trajectory points using sort_idx
    ksomething_perm = torch.zeros_like(ksomething)
    ksomething_perm[sort_idx, :] = ksomething
    return ksomething_perm


def test_ktraj_raw_reshape(cartesian_grid):
    """Test reshaping between raw ktrajectory and ktrajectory."""
    # Create Cartesian raw trajectory
    nk0 = 10
    nk1 = 20
    nk2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(nk2, nk1, nk0, jitter=0.0)

    # Repeat other dimension (e.g. multiple slices)
    nother = 3
    kz_full = repeat(kz_full, '1 k2 k1 k0->other k2 k1 k0', other=nother)
    ky_full = repeat(ky_full, '1 k2 k1 k0->other k2 k1 k0', other=nother)
    kx_full = repeat(kx_full, '1 k2 k1 k0->other k2 k1 k0', other=nother)

    # Reshape and repeat from (nother nk2 nk1 nk0) to (nother*nk2*nk1 nk0) and permute randomly
    sort_idx = np.random.permutation(np.linspace(0, nother * nk2 * nk1 - 1, nother * nk2 * nk1))
    kz_raw = make_ktraj_raw_shape(kz_full, sort_idx)
    ky_raw = make_ktraj_raw_shape(ky_full, sort_idx)
    kx_raw = make_ktraj_raw_shape(kx_full, sort_idx)

    # Create raw trajectory
    ktraj_raw = KTrajectoryRawShape(kz_raw, ky_raw, kx_raw)

    # Reshape to original trajectory
    ktraj = ktraj_raw.reshape(sort_idx, nk2, nk1, repeat_detection_tolerance=None)

    # Compare trajectories
    torch.testing.assert_close(ktraj.kz, kz_full)
    torch.testing.assert_close(ktraj.ky, ky_full)
    torch.testing.assert_close(ktraj.kx, kx_full)
