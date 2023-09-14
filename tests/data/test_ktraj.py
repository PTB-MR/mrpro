"""Tests for KTrajectory classes."""

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

from mrpro.data import KTrajectoryRpe
from mrpro.data import KTrajectorySunflowerGoldenRpe
from tests.data.conftest import random_kheader


@pytest.fixture(scope='function')
def valid_rpe_kheader(monkeypatch, random_kheader):
    """KHeader with all necessary parameters for RPE trajectories."""
    # K-space dimensions
    nk0 = 200
    nk1 = 20
    nk2 = 10

    # List of k1 and k2 indices in the shape (d4, k2, k1)
    k1 = torch.linspace(0, nk1 - 1, nk1, dtype=torch.int32)
    k2 = torch.linspace(0, nk2 - 1, nk2, dtype=torch.int32)
    idx_k1, idx_k2 = torch.meshgrid(k1, k2)
    idx_k1 = torch.reshape(idx_k1, (1, nk2, nk1))
    idx_k2 = torch.reshape(idx_k2, (1, nk2, nk1))

    # Set parameters for RPE trajectory
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1) + nk0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1) + nk0 // 2)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k1', idx_k1)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k2', idx_k2)
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'center', int(nk1 // 2))
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'max', int(nk1 - 1))
    monkeypatch.setattr(random_kheader.encoding_limits.k2, 'center', int(nk2 // 2))
    return random_kheader


def rpe_traj_shape(valid_rpe_kheader):
    """Expected shape of trajectory based on KHeader."""
    nk0 = valid_rpe_kheader.acq_info.number_of_samples[0, 0, 0]
    nk1 = valid_rpe_kheader.acq_info.idx.k1.shape[2]
    nk2 = valid_rpe_kheader.acq_info.idx.k1.shape[1]
    return torch.Size([1, 3, nk2, nk1, nk0])


def test_KTrajectoryRpe(valid_rpe_kheader):
    """Calculate RPE trajectory."""
    ktrajectory = KTrajectoryRpe(angle=np.pi * 0.618034)
    ktraj = ktrajectory.calc_traj(valid_rpe_kheader)
    assert ktraj.shape == rpe_traj_shape(valid_rpe_kheader)


def test_KTrajectorySunflowerGoldenRpe(valid_rpe_kheader):
    """Calculate RPE Sunflower trajectory."""
    ktrajectory = KTrajectorySunflowerGoldenRpe(rad_us_factor=1)
    ktraj = ktrajectory.calc_traj(valid_rpe_kheader)
    assert ktraj.shape == rpe_traj_shape(valid_rpe_kheader)
