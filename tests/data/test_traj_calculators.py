"""Tests for KTrajectory Calculator classes."""

import pytest
import torch
from einops import rearrange
from mrpro.data import KData
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
    k1_idx = torch.arange(n_k1)[:, None]

    trajectory_calculator = KTrajectoryRadial2D()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
    )

    assert trajectory.kz.shape == (1, 1, 1, 1)
    assert trajectory.ky.shape == (1, 1, n_k1, n_k0)
    assert trajectory.kx.shape == (1, 1, n_k1, n_k0)


def test_KTrajectoryRpe():
    """Test shapes returned by KTrajectoryRpe"""
    n_k0 = 100
    n_k1 = 20
    n_k2 = 10
    k2_idx = torch.arange(n_k2)[:, None, None]
    k1_idx = torch.arange(n_k1)[:, None]

    trajectory_calculator = KTrajectoryRpe(angle=torch.pi * 0.618034)
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
    )

    assert trajectory.kz.shape == (1, n_k2, n_k1, 1)
    assert trajectory.ky.shape == (1, n_k2, n_k1, 1)
    assert trajectory.kx.shape == (1, 1, 1, n_k0)


def test_KTrajectoryRpe_angle():
    """Test that every second line matches a trajectory with double the angular gap."""
    n_k0 = 100
    n_k1 = 20
    n_k2 = 10
    k2_idx = torch.arange(n_k2)[:, None, None]
    k1_idx = torch.arange(n_k1)[:, None]
    angle = torch.pi / n_k2

    trajectory1_calculator = KTrajectoryRpe(angle=angle, shift_between_rpe_lines=(0,))
    trajectory1 = trajectory1_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
    )
    tensor1 = trajectory1.as_tensor()

    # trajectory with double the angular gap
    trajectory2_calculator = KTrajectoryRpe(
        angle=2 * angle,
        shift_between_rpe_lines=torch.tensor([0, 0, 0, 0]),
    )
    trajectory2 = trajectory2_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
    )
    tensor2 = trajectory2.as_tensor()

    torch.testing.assert_close(tensor1[..., ::2, :, :], tensor2[..., : n_k2 // 2, :, :])


def test_KTrajectorySunflowerGoldenRpe():
    """Test shape returned by KTrajectorySunflowerGoldenRpe"""
    n_k0 = 100
    n_k1 = 20
    n_k2 = 10
    k2_idx = torch.arange(n_k2)[:, None]
    k1_idx = torch.arange(n_k1)

    trajectory_calculator = KTrajectorySunflowerGoldenRpe()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
    )

    assert trajectory.broadcasted_shape == (1, n_k2, n_k1, n_k0)


def test_KTrajectoryCartesian():
    """Calculate Cartesian trajectory."""
    n_k0 = 30
    n_k1 = 20
    n_k2 = 10
    k2_idx = torch.arange(n_k2)[:, None, None]
    k1_idx = torch.arange(n_k1)[:, None]

    trajectory_calculator = KTrajectoryCartesian()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
        k2_center=n_k2 // 2,
    )
    assert trajectory.kz.shape == (1, n_k2, 1, 1)
    assert trajectory.ky.shape == (1, 1, n_k1, 1)
    assert trajectory.kx.shape == (1, 1, 1, n_k0)


def test_KTrajectoryCartesian_bipolar():
    """Verify that the readout for the second part of a bipolar readout is reversed"""
    n_k0 = 30
    n_k1 = 20
    n_k2 = 10
    k2_idx = torch.arange(n_k2)[:, None, None]
    k1_idx = torch.arange(n_k1)[:, None]

    reversed_readout_mask = torch.zeros(n_k1, dtype=torch.bool)
    reversed_readout_mask[1] = True

    trajectory_calculator = KTrajectoryCartesian()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
        k2_center=n_k2 // 2,
        reversed_readout_mask=reversed_readout_mask,
    )

    torch.testing.assert_close(trajectory.kx[..., 0, :], torch.flip(trajectory.kx[..., 1, :], dims=(-1,)))


def test_KTrajectoryIsmrmrdRadial(ismrmrd_rad):
    """Verify ismrmrd trajectory."""
    # Calculate trajectory based on header information
    angle = torch.pi / (ismrmrd_rad.matrix_size // ismrmrd_rad.acceleration)
    kdata = KData.from_file(ismrmrd_rad.filename, KTrajectoryRadial2D(angle=angle))
    trajectory_calc = kdata.traj.as_tensor()

    # Read trajectory from raw data file
    kdata = KData.from_file(ismrmrd_rad.filename, KTrajectoryIsmrmrd())
    trajectory_read = kdata.traj.as_tensor()

    torch.testing.assert_close(trajectory_calc, trajectory_read, atol=1e-2, rtol=1e-3)


@pytest.fixture(scope='session')
def pulseq_example_rad_seq(tmp_path_factory):
    seq_filename = tmp_path_factory.mktemp('mrpro') / 'radial.seq'
    seq = PulseqRadialTestSeq(seq_filename, n_k0=256, n_spokes=10)
    return seq


def test_KTrajectoryPulseq(pulseq_example_rad_seq):
    """Test pulseq File reader with valid seq File."""
    # TODO: Test with invalid seq file
    trajectory_calculator = KTrajectoryPulseq(seq_path=pulseq_example_rad_seq.seq_filename)
    trajectory = trajectory_calculator(
        n_k0=pulseq_example_rad_seq.n_k0, encoding_matrix=pulseq_example_rad_seq.encoding_matrix
    )

    kx_test = rearrange(pulseq_example_rad_seq.traj_analytical.kx, 'other k2 k1 k0 -> (other k2 k1) 1 1 k0')
    kx_test = kx_test * pulseq_example_rad_seq.encoding_matrix.x / (2 * kx_test.abs().max())
    ky_test = rearrange(pulseq_example_rad_seq.traj_analytical.ky, 'other k2 k1 k0 -> (other k2 k1) 1 1 k0')
    ky_test = ky_test * pulseq_example_rad_seq.encoding_matrix.y / (2 * ky_test.abs().max())

    torch.testing.assert_close(trajectory.kx.to(torch.float32), kx_test.to(torch.float32), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(trajectory.ky.to(torch.float32), ky_test.to(torch.float32), atol=1e-2, rtol=1e-3)
