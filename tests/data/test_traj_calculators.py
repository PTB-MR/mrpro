"""Tests for KTrajectory Calculator classes."""

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

from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.data.traj_calculators import KTrajectoryRadial2D
from mrpro.data.traj_calculators import KTrajectoryRpe
from mrpro.data.traj_calculators import KTrajectorySunflowerGoldenRpe
from mrpro.data.traj_calculators._KTrajectoryPulseq import KTrajectoryPulseq
from tests.conftest import random_kheader
from tests.data import IsmrmrdRawTestData
from tests.data._PulseqRadialTestSeq import PulseqRadialTestSeq
from tests.phantoms.test_ellipse_phantom import ph_ellipse


@pytest.fixture(scope='function')
def valid_rad2d_kheader(monkeypatch, random_kheader):
    """KHeader with all necessary parameters for radial 2D trajectories."""
    # K-space dimensions
    nk0 = 200
    nk1 = 20

    # List of k1 indices in the shape
    idx_k1 = torch.arange(nk1, dtype=torch.int32)[None, None, ...]

    # Set parameters for radial 2D trajectory
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1) + nk0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1) + nk0 // 2)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k1', idx_k1)

    return random_kheader


def radial2D_traj_shape(valid_rad2d_kheader):
    """Expected shape of trajectory based on KHeader."""
    nk0 = valid_rad2d_kheader.acq_info.number_of_samples[0, 0, 0]
    nk1 = valid_rad2d_kheader.acq_info.idx.k1.shape[2]
    nk2 = 1
    nother = 1
    return (torch.Size([nother, 1, 1, 1]), torch.Size([nother, nk2, nk1, nk0]), torch.Size([nother, nk2, nk1, nk0]))


def test_KTrajectoryRadial2D_golden(valid_rad2d_kheader):
    """Calculate Radial 2D trajectory with golden angle."""
    ktrajectory = KTrajectoryRadial2D(angle=torch.pi * 0.618034)
    ktraj = ktrajectory(valid_rad2d_kheader)
    valid_shape = radial2D_traj_shape(valid_rad2d_kheader)
    assert ktraj.kx.shape == valid_shape[2]
    assert ktraj.ky.shape == valid_shape[1]
    assert ktraj.kz.shape == valid_shape[0]


from tests.data.test_kdata import ismrmrd_cart


@pytest.fixture(scope='function')
def valid_rpe_kheader(monkeypatch, random_kheader):
    """KHeader with all necessary parameters for RPE trajectories."""
    # K-space dimensions
    nk0 = 200
    nk1 = 20
    nk2 = 10

    # List of k1 and k2 indices in the shape (other, k2, k1)
    k1 = torch.linspace(0, nk1 - 1, nk1, dtype=torch.int32)
    k2 = torch.linspace(0, nk2 - 1, nk2, dtype=torch.int32)
    idx_k1, idx_k2 = torch.meshgrid(k1, k2, indexing='xy')
    idx_k1 = torch.reshape(idx_k1, (1, nk2, nk1))
    idx_k2 = torch.reshape(idx_k2, (1, nk2, nk1))

    # Set parameters for RPE trajectory
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1) + nk0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1) + nk0 // 2)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k1', idx_k1)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k2', idx_k2)
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'center', int(nk1 // 2))
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'max', int(nk1 - 1))
    return random_kheader


@pytest.fixture(scope='function')
def valid_radial_kheader(monkeypatch, random_kheader):
    """KHeader with all necessary parameters for radial trajectories."""
    # K-space dimensions
    nk0 = 256
    nk1 = 10
    nk2 = 1

    # List of k1 and k2 indices in the shape (other, k2, k1)
    k1 = torch.linspace(0, nk1 - 1, nk1, dtype=torch.int32)
    k2 = torch.linspace(0, nk2 - 1, nk2, dtype=torch.int32)
    idx_k1, idx_k2 = torch.meshgrid(k1, k2, indexing='xy')
    idx_k1 = torch.reshape(idx_k1, (1, nk2, nk1))
    idx_k2 = torch.reshape(idx_k2, (1, nk2, nk1))

    # Set parameters for RPE trajectory
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1) + nk0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1) + nk0 // 2)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k1', idx_k1)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k2', idx_k2)
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'center', int(nk1 // 2))
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'max', int(nk1 - 1))
    return random_kheader


def rpe_traj_shape(valid_rpe_kheader):
    """Expected shape of trajectory based on KHeader."""
    nk0 = valid_rpe_kheader.acq_info.number_of_samples[0, 0, 0]
    nk1 = valid_rpe_kheader.acq_info.idx.k1.shape[2]
    nk2 = valid_rpe_kheader.acq_info.idx.k1.shape[1]
    nother = 1
    return (torch.Size([nother, nk2, nk1, 1]), torch.Size([nother, nk2, nk1, 1]), torch.Size([nother, 1, 1, nk0]))


def test_KTrajectoryRpe_golden(valid_rpe_kheader):
    """Calculate RPE trajectory with golden angle."""
    ktrajectory = KTrajectoryRpe(angle=torch.pi * 0.618034)
    ktraj = ktrajectory(valid_rpe_kheader)
    valid_shape = rpe_traj_shape(valid_rpe_kheader)
    assert ktraj.kz.shape == valid_shape[0]
    assert ktraj.ky.shape == valid_shape[1]
    assert ktraj.kx.shape == valid_shape[2]


def test_KTrajectoryRpe_uniform(valid_rpe_kheader):
    """Calculate RPE trajectory with uniform angle."""
    num_rpe_lines = valid_rpe_kheader.acq_info.idx.k1.shape[1]
    ktrajectory1 = KTrajectoryRpe(angle=torch.pi / num_rpe_lines, shift_between_rpe_lines=torch.tensor([0]))
    ktraj1 = ktrajectory1(valid_rpe_kheader)
    # Calculate trajectory with half the angular gap such that every second line should be the same as above
    ktrajectory2 = KTrajectoryRpe(angle=torch.pi / (2 * num_rpe_lines), shift_between_rpe_lines=torch.tensor([0]))
    ktraj2 = ktrajectory2(valid_rpe_kheader)

    torch.testing.assert_close(ktraj1.kx[:, : num_rpe_lines // 2, :, :], ktraj2.kx[:, ::2, :, :])
    torch.testing.assert_close(ktraj1.ky[:, : num_rpe_lines // 2, :, :], ktraj2.ky[:, ::2, :, :])
    torch.testing.assert_close(ktraj1.kz[:, : num_rpe_lines // 2, :, :], ktraj2.kz[:, ::2, :, :])


def test_KTrajectoryRpe_shift(valid_rpe_kheader):
    """Evaluate radial shifts for RPE trajectory."""
    ktrajectory1 = KTrajectoryRpe(angle=torch.pi * 0.618034, shift_between_rpe_lines=torch.tensor([0.25]))
    ktraj1 = ktrajectory1(valid_rpe_kheader)
    ktrajectory2 = KTrajectoryRpe(
        angle=torch.pi * 0.618034, shift_between_rpe_lines=torch.tensor([0.25, 0.25, 0.25, 0.25])
    )
    ktraj2 = ktrajectory2(valid_rpe_kheader)
    torch.testing.assert_close(ktraj1.as_tensor(), ktraj2.as_tensor())


def test_KTrajectorySunflowerGoldenRpe(valid_rpe_kheader):
    """Calculate RPE Sunflower trajectory."""
    ktrajectory = KTrajectorySunflowerGoldenRpe(rad_us_factor=2)
    ktraj = ktrajectory(valid_rpe_kheader)
    assert ktraj.broadcasted_shape == np.broadcast_shapes(*rpe_traj_shape(valid_rpe_kheader))


@pytest.fixture(scope='session')
def ismrmrd_rad(ph_ellipse, tmp_path_factory):
    """Data set with uniform radial k-space sampling."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_rad.h5'
    ismrmrd_kdat = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        phantom=ph_ellipse.phantom,
        trajectory_type='radial',
        acceleration=4,
    )
    return ismrmrd_kdat


def test_KTrajectoryIsmrmrdRadial(ismrmrd_rad):
    """Verify ismrmrd trajectory."""
    # Calculate trajectory based on header information
    angle_step = torch.pi / (ismrmrd_rad.matrix_size // ismrmrd_rad.acceleration)
    k = KData.from_file(ismrmrd_rad.filename, KTrajectoryRadial2D(angle=angle_step))
    ktraj_calc = k.traj.as_tensor()

    # Read trajectory from raw data file
    k = KData.from_file(ismrmrd_rad.filename, KTrajectoryIsmrmrd())
    ktraj_read = k.traj.as_tensor()

    torch.testing.assert_close(ktraj_calc, ktraj_read)


def pulseqtestdata(tmp_path_factory):
    seq_filename = tmp_path_factory.mktemp('mrpro') / 'radial.seq'
    seq = PulseqRadialTestSeq(str(seq_filename), Nx=256, Nspokes=10)
    return seq


def test_KTrajectoryPulseq_validseq_random_header(pulseqtestdata, valid_radial_kheader):
    """Test pulseq File reader with valid seq File."""
    # TODO: Test with valid header
    # TODO: Test with invalid seq file

    ktrajectory = KTrajectoryPulseq(path=pulseqtestdata.seq_filename)
    traj = ktrajectory(kheader=valid_radial_kheader)

    kx_test = pulseqtestdata.traj_analytical.kx.squeeze(0).squeeze(0)
    kx_test = kx_test / torch.max(torch.abs(kx_test)) * torch.pi

    torch.testing.assert_close(traj.kx, kx_test, atol=5e-4, rtol=1e-3)
