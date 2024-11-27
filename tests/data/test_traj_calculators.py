"""Tests for KTrajectory Calculator classes."""

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


def test_KTrajectoryRadial2D():
    """Test shapes returned by KTrajectoryRadial2D."""

    n_k0 = 256
    n_k1 = 10

    trajectory_calculator = KTrajectoryRadial2D()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=torch.arange(n_k1)[None, None, :, None],
    )
    assert trajectory.kz.shape == (1, 1, 1, 1)
    assert trajectory.ky.shape == (1, 1, n_k1, n_k0)
    assert trajectory.kx.shape == (1, 1, n_k1, n_k0)


def test_KTrajectoryRpe():
    """Test shapes returned by KTrajectoryRpe"""
    n_k0 = 100
    n_k1 = 20
    n_k2 = 10

    trajectory_calculator = KTrajectoryRpe(angle=torch.pi * 0.618034)
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=torch.arange(n_k2)[None, :, None, None],
        k1_center=n_k1 // 2,
        k2_idx=torch.arange(n_k1)[None, None, :, None],
    )
    assert trajectory.kz.shape == (1, n_k2, n_k1, 1)
    assert trajectory.ky.shape == (1, n_k2, n_k1, 1)
    assert trajectory.kx.shape == (1, 1, 1, n_k0)


def test_KTrajectoryRpe_angle():
    """Test that every second line matches the first half of lines of a trajectory with double the angular gap."""
    n_k0 = 100
    n_k1 = 20
    n_k2 = 10
    k1_idx = torch.arange(n_k1)[None, None, :, None]
    k2_idx = torch.arange(n_k2)[None, :, None, None]
    trajectory1_calculator = KTrajectoryRpe(angle=torch.pi / n_k1, shift_between_rpe_lines=(0,))
    trajectory1 = trajectory1_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
    )
    # Calculate trajectory with half the angular gap such that every second line should be the same as above
    trajectory2_calculator = KTrajectoryRpe(
        angle=torch.pi / (2 * n_k1),
        shift_between_rpe_lines=torch.tensor([0, 0, 0, 0]),
    )
    trajectory2 = trajectory2_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
    )

    torch.testing.assert_close(trajectory1.kx[:, : n_k1 // 2, :, :], trajectory2.kx[:, ::2, :, :])
    torch.testing.assert_close(trajectory1.ky[:, : n_k1 // 2, :, :], trajectory2.ky[:, ::2, :, :])
    torch.testing.assert_close(trajectory1.kz[:, : n_k1 // 2, :, :], trajectory2.kz[:, ::2, :, :])


def test_KTrajectorySunflowerGoldenRpe():
    """Test shape returned by KTrajectorySunflowerGoldenRpe"""
    n_k0 = 100
    n_k1 = 20
    n_k2 = 10
    k1_idx = torch.arange(n_k1)[None, None, :, None]
    k2_idx = torch.arange(n_k2)[None, :, None, None]
    trajectory_calculator = KTrajectorySunflowerGoldenRpe()
    trajectory = trajectory_calculator(
        n_k0=n_k0, k0_center=n_k0 // 2, k1_idx=k1_idx, k1_center=n_k1 // 2, k2_idx=k2_idx
    )
    assert trajectory.broadcasted_shape == (1, n_k2, n_k1, n_k0)


def test_KTrajectoryCartesian(valid_cartesian_kheader):
    """Calculate Cartesian trajectory."""
    n_k0 = 30
    n_k1 = 20
    n_k2 = 10
    trajectory_calculator = KTrajectoryCartesian()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=torch.arange(n_k1)[None, None, :, None],
        k1_center=n_k1 // 2,
        k2_idx=torch.arange(n_k2)[None, :, None, None],
    )
    assert trajectory.kz.shape == (1, n_k2, 1, 1)
    assert trajectory.ky.shape == (1, 1, n_k1, 1)
    assert trajectory.kx.shape == (1, 1, 1, n_k0)


def test_KTrajectoryCartesian_bipolar(valid_cartesian_kheader_bipolar):
    """Verify that the readout for the second part of a bipolar readout is reversed"""
    trajectory_calculator = KTrajectoryCartesian()
    n_k0 = 30
    n_k1 = 20
    n_k2 = 10
    reversed_readout_mask = torch.zeros(n_k1, 1, dtype=torch.bool)
    reversed_readout_mask[1] = True
    trajectory_calculator = KTrajectoryCartesian()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=torch.arange(n_k1)[None, None, :, None],
        k1_center=n_k1 // 2,
        k2_idx=torch.arange(n_k2)[None, :, None, None],
        reversed_readout_mask=reversed_readout_mask,
    )
    torch.testing.assert_close(trajectory.kx[..., 0, :], torch.flip(trajectory.kx[..., 1, :], dims=(-1,)))


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


def test_KTrajectoryPulseq(pulseq_example_rad_seq, valid_rad2d_kheader):
    """Test pulseq File reader with valid seq File."""
    # TODO: Test with invalid seq file

    trajectory_calculator = KTrajectoryPulseq(seq_path=pulseq_example_rad_seq.seq_filename)
    trajectory = trajectory_calculator(n_k0=n_k0, encoding_matrix=encoding_matrix)

    kx_test = pulseq_example_rad_seq.traj_analytical.kx.squeeze(0).squeeze(0)
    kx_test *= valid_rad2d_kheader.encoding_matrix.x / (2 * torch.max(torch.abs(kx_test)))

    ky_test = pulseq_example_rad_seq.traj_analytical.ky.squeeze(0).squeeze(0)
    ky_test *= valid_rad2d_kheader.encoding_matrix.y / (2 * torch.max(torch.abs(ky_test)))

    torch.testing.assert_close(trajectory.kx.to(torch.float32), kx_test.to(torch.float32), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(trajectory.ky.to(torch.float32), ky_test.to(torch.float32), atol=1e-2, rtol=1e-3)


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
    return


@pytest.fixture
def valid_cartesian_kheader_bipolar(monkeypatch, valid_cartesian_kheader):
    """Set readout of other==1 to reversed."""
    acq_info_flags = valid_cartesian_kheader.acq_info.flags
    acq_info_flags[1, ...] = AcqFlags.ACQ_IS_REVERSE.value
    monkeypatch.setattr(valid_cartesian_kheader.acq_info, 'flags', acq_info_flags)
    return valid_cartesian_kheader
