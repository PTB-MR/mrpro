"""Tests for KTrajectory Calculator classes."""

import numpy as np
import pytest
import torch
from einops import repeat
from mrpro.data import KData
from mrpro.data.enums import AcqFlags
from mrpro.data.traj_calculators import (
    KTrajectoryCartesian,
    KTrajectoryIsmrmrd,
    KTrajectoryPulseq,
    KTrajectoryRadial2D,
    KTrajectoryRpe,
    KTrajectorySunflowerGoldenRpe,
)

from tests.data import IsmrmrdRawTestData, PulseqRadialTestSeq


@pytest.fixture
def valid_rad2d_kheader(monkeypatch, random_kheader):
    """KHeader with all necessary parameters for radial 2D trajectories."""
    # K-space dimensions
    n_k0 = 256
    n_k1 = 10
    n_k2 = 1

    # List of k1 indices in the shape
    idx_k1 = repeat(torch.arange(n_k1, dtype=torch.int32), 'k1 -> other k2 k1', other=1, k2=1)

    # Set parameters for radial 2D trajectory (AcqInfo is of shape (other k2 k1 dim=1 or 3))
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1)[..., None] + n_k0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1)[..., None] + n_k0 // 2)
    monkeypatch.setattr(random_kheader.acq_info, 'flags', torch.zeros_like(idx_k1)[..., None])
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k1', idx_k1)

    # This is only needed for Pulseq trajectory calculation
    monkeypatch.setattr(random_kheader.encoding_matrix, 'x', n_k0)
    monkeypatch.setattr(random_kheader.encoding_matrix, 'y', n_k0)  # square encoding in kx-ky plane
    monkeypatch.setattr(random_kheader.encoding_matrix, 'z', n_k2)

    return random_kheader


def radial2D_traj_shape(valid_rad2d_kheader):
    """Expected shape of trajectory based on KHeader."""
    n_k0 = valid_rad2d_kheader.acq_info.number_of_samples[0, 0, 0]
    n_k1 = valid_rad2d_kheader.acq_info.idx.k1.shape[2]
    n_k2 = 1
    n_other = 1
    return (
        torch.Size([n_other, 1, 1, 1]),
        torch.Size([n_other, n_k2, n_k1, n_k0]),
        torch.Size([n_other, n_k2, n_k1, n_k0]),
    )


def test_KTrajectoryRadial2D_golden(valid_rad2d_kheader):
    """Calculate Radial 2D trajectory with golden angle."""
    trajectory_calculator = KTrajectoryRadial2D(angle=torch.pi * 0.618034)
    trajectory = trajectory_calculator(valid_rad2d_kheader)
    valid_shape = radial2D_traj_shape(valid_rad2d_kheader)
    assert trajectory.kx.shape == valid_shape[2]
    assert trajectory.ky.shape == valid_shape[1]
    assert trajectory.kz.shape == valid_shape[0]


@pytest.fixture
def valid_rpe_kheader(monkeypatch, random_kheader):
    """KHeader with all necessary parameters for RPE trajectories."""
    # K-space dimensions
    n_k0 = 200
    n_k1 = 20
    n_k2 = 10

    # List of k1 and k2 indices in the shape (other, k2, k1)
    k1 = torch.linspace(0, n_k1 - 1, n_k1, dtype=torch.int32)
    k2 = torch.linspace(0, n_k2 - 1, n_k2, dtype=torch.int32)
    idx_k1, idx_k2 = torch.meshgrid(k1, k2, indexing='xy')
    idx_k1 = torch.reshape(idx_k1, (1, n_k2, n_k1))
    idx_k2 = torch.reshape(idx_k2, (1, n_k2, n_k1))

    # Set parameters for RPE trajectory (AcqInfo is of shape (other k2 k1 dim=1 or 3))
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1)[..., None] + n_k0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1)[..., None] + n_k0 // 2)
    monkeypatch.setattr(random_kheader.acq_info, 'flags', torch.zeros_like(idx_k1)[..., None])
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k1', idx_k1)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k2', idx_k2)
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'center', int(n_k1 // 2))
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'max', int(n_k1 - 1))
    return random_kheader


def rpe_traj_shape(valid_rpe_kheader):
    """Expected shape of trajectory based on KHeader."""
    n_k0 = valid_rpe_kheader.acq_info.number_of_samples[0, 0, 0]
    n_k1 = valid_rpe_kheader.acq_info.idx.k1.shape[2]
    n_k2 = valid_rpe_kheader.acq_info.idx.k1.shape[1]
    n_other = 1
    return (
        torch.Size([n_other, n_k2, n_k1, 1]),
        torch.Size([n_other, n_k2, n_k1, 1]),
        torch.Size([n_other, 1, 1, n_k0]),
    )


def test_KTrajectoryRpe_golden(valid_rpe_kheader):
    """Calculate RPE trajectory with golden angle."""
    trajectory_calculator = KTrajectoryRpe(angle=torch.pi * 0.618034)
    trajectory = trajectory_calculator(valid_rpe_kheader)
    valid_shape = rpe_traj_shape(valid_rpe_kheader)
    assert trajectory.kz.shape == valid_shape[0]
    assert trajectory.ky.shape == valid_shape[1]
    assert trajectory.kx.shape == valid_shape[2]


def test_KTrajectoryRpe_uniform(valid_rpe_kheader):
    """Calculate RPE trajectory with uniform angle."""
    n_rpe_lines = valid_rpe_kheader.acq_info.idx.k1.shape[1]
    trajectory1_calculator = KTrajectoryRpe(angle=torch.pi / n_rpe_lines, shift_between_rpe_lines=torch.tensor([0]))
    trajectory1 = trajectory1_calculator(valid_rpe_kheader)
    # Calculate trajectory with half the angular gap such that every second line should be the same as above
    trajectory2_calculator = KTrajectoryRpe(
        angle=torch.pi / (2 * n_rpe_lines),
        shift_between_rpe_lines=torch.tensor([0]),
    )
    trajectory2 = trajectory2_calculator(valid_rpe_kheader)

    torch.testing.assert_close(trajectory1.kx[:, : n_rpe_lines // 2, :, :], trajectory2.kx[:, ::2, :, :])
    torch.testing.assert_close(trajectory1.ky[:, : n_rpe_lines // 2, :, :], trajectory2.ky[:, ::2, :, :])
    torch.testing.assert_close(trajectory1.kz[:, : n_rpe_lines // 2, :, :], trajectory2.kz[:, ::2, :, :])


def test_KTrajectoryRpe_shift(valid_rpe_kheader):
    """Evaluate radial shifts for RPE trajectory."""
    trajectory1_calculator = KTrajectoryRpe(angle=torch.pi * 0.618034, shift_between_rpe_lines=torch.tensor([0.25]))
    trajectory1 = trajectory1_calculator(valid_rpe_kheader)
    trajectory2_calculator = KTrajectoryRpe(
        angle=torch.pi * 0.618034,
        shift_between_rpe_lines=torch.tensor([0.25, 0.25, 0.25, 0.25]),
    )
    trajectory2 = trajectory2_calculator(valid_rpe_kheader)
    torch.testing.assert_close(trajectory1.as_tensor(), trajectory2.as_tensor())


def test_KTrajectorySunflowerGoldenRpe(valid_rpe_kheader):
    """Calculate RPE Sunflower trajectory."""
    trajectory_calculator = KTrajectorySunflowerGoldenRpe(rad_us_factor=2)
    trajectory = trajectory_calculator(valid_rpe_kheader)
    assert trajectory.broadcasted_shape == np.broadcast_shapes(*rpe_traj_shape(valid_rpe_kheader))


@pytest.fixture
def valid_cartesian_kheader(monkeypatch, random_kheader):
    """KHeader with all necessary parameters for Cartesian trajectories."""
    # K-space dimensions
    n_k0 = 200
    n_k1 = 20
    n_k2 = 10
    n_other = 2

    # List of k1 and k2 indices in the shape (other, k2, k1)
    k1 = torch.linspace(0, n_k1 - 1, n_k1, dtype=torch.int32)
    k2 = torch.linspace(0, n_k2 - 1, n_k2, dtype=torch.int32)
    idx_k1, idx_k2 = torch.meshgrid(k1, k2, indexing='xy')
    idx_k1 = repeat(torch.reshape(idx_k1, (n_k2, n_k1)), 'k2 k1->other k2 k1', other=n_other)
    idx_k2 = repeat(torch.reshape(idx_k2, (n_k2, n_k1)), 'k2 k1->other k2 k1', other=n_other)

    # Set parameters for Cartesian trajectory (AcqInfo is of shape (other k2 k1 dim=1 or 3))
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1)[..., None] + n_k0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1)[..., None] + n_k0 // 2)
    monkeypatch.setattr(random_kheader.acq_info, 'flags', torch.zeros_like(idx_k1)[..., None])
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k1', idx_k1)
    monkeypatch.setattr(random_kheader.acq_info.idx, 'k2', idx_k2)
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'center', int(n_k1 // 2))
    monkeypatch.setattr(random_kheader.encoding_limits.k1, 'max', int(n_k1 - 1))
    monkeypatch.setattr(random_kheader.encoding_limits.k2, 'center', int(n_k2 // 2))
    monkeypatch.setattr(random_kheader.encoding_limits.k2, 'max', int(n_k2 - 1))
    return random_kheader


def cartesian_traj_shape(valid_cartesian_kheader):
    """Expected shape of trajectory based on KHeader."""
    n_k0 = valid_cartesian_kheader.acq_info.number_of_samples[0, 0, 0]
    n_k1 = valid_cartesian_kheader.acq_info.idx.k1.shape[2]
    n_k2 = valid_cartesian_kheader.acq_info.idx.k1.shape[1]
    n_other = 1  # trajectory along other is the same
    return (torch.Size([n_other, n_k2, 1, 1]), torch.Size([n_other, 1, n_k1, 1]), torch.Size([n_other, 1, 1, n_k0]))


def test_KTrajectoryCartesian(valid_cartesian_kheader):
    """Calculate Cartesian trajectory."""
    trajectory_calculator = KTrajectoryCartesian()
    trajectory = trajectory_calculator(valid_cartesian_kheader)
    valid_shape = cartesian_traj_shape(valid_cartesian_kheader)
    assert trajectory.kz.shape == valid_shape[0]
    assert trajectory.ky.shape == valid_shape[1]
    assert trajectory.kx.shape == valid_shape[2]


@pytest.fixture
def valid_cartesian_kheader_bipolar(monkeypatch, valid_cartesian_kheader):
    """Set readout of other==1 to reversed."""
    acq_info_flags = valid_cartesian_kheader.acq_info.flags
    acq_info_flags[1, ...] = AcqFlags.ACQ_IS_REVERSE.value
    monkeypatch.setattr(valid_cartesian_kheader.acq_info, 'flags', acq_info_flags)
    return valid_cartesian_kheader


def test_KTrajectoryCartesian_bipolar(valid_cartesian_kheader_bipolar):
    """Calculate Cartesian trajectory with bipolar readout."""
    trajectory_calculator = KTrajectoryCartesian()
    trajectory = trajectory_calculator(valid_cartesian_kheader_bipolar)
    # Verify that the readout for the second part of the bipolar readout is reversed
    torch.testing.assert_close(trajectory.kx[0, ...], torch.flip(trajectory.kx[1, ...], dims=(-1,)))


@pytest.fixture(scope='session')
def ismrmrd_rad(ellipse_phantom, tmp_path_factory):
    """Data set with uniform radial k-space sampling."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_rad.h5'
    ismrmrd_data = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        phantom=ellipse_phantom.phantom,
        trajectory_type='radial',
        acceleration=4,
    )
    return ismrmrd_data


def test_KTrajectoryIsmrmrdRadial(ismrmrd_rad):
    """Verify ismrmrd trajectory."""
    # Calculate trajectory based on header information
    angle_step = torch.pi / (ismrmrd_rad.matrix_size // ismrmrd_rad.acceleration)
    kdata = KData.from_file(ismrmrd_rad.filename, KTrajectoryRadial2D(angle=angle_step))
    trajectory_calc = kdata.traj.as_tensor()

    # Read trajectory from raw data file
    kdata = KData.from_file(ismrmrd_rad.filename, KTrajectoryIsmrmrd())
    trajectory_read = kdata.traj.as_tensor()

    torch.testing.assert_close(trajectory_calc, trajectory_read, atol=1e-2, rtol=1e-3)


@pytest.fixture(scope='session')
def pulseq_example_rad_seq(tmp_path_factory):
    seq_filename = tmp_path_factory.mktemp('mrpro') / 'radial.seq'
    seq = PulseqRadialTestSeq(seq_filename, n_x=256, n_spokes=10)
    return seq


def test_KTrajectoryPulseq_validseq_random_header(pulseq_example_rad_seq, valid_rad2d_kheader):
    """Test pulseq File reader with valid seq File."""
    # TODO: Test with valid header
    # TODO: Test with invalid seq file

    trajectory_calculator = KTrajectoryPulseq(seq_path=pulseq_example_rad_seq.seq_filename)
    trajectory = trajectory_calculator(kheader=valid_rad2d_kheader)

    kx_test = pulseq_example_rad_seq.traj_analytical.kx.squeeze(0).squeeze(0)
    kx_test *= valid_rad2d_kheader.encoding_matrix.x / (2 * torch.max(torch.abs(kx_test)))

    ky_test = pulseq_example_rad_seq.traj_analytical.ky.squeeze(0).squeeze(0)
    ky_test *= valid_rad2d_kheader.encoding_matrix.y / (2 * torch.max(torch.abs(ky_test)))

    torch.testing.assert_close(trajectory.kx.to(torch.float32), kx_test.to(torch.float32), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(trajectory.ky.to(torch.float32), ky_test.to(torch.float32), atol=1e-2, rtol=1e-3)
