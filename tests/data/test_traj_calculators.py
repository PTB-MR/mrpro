"""Tests for KTrajectory Calculator classes."""

import pytest
import torch
from einops import rearrange
from mr2.data import KData, SpatialDimension
from mr2.data.traj_calculators import (
    KTrajectoryCartesian,
    KTrajectoryIsmrmrd,
    KTrajectoryPulseq,
    KTrajectoryRadial2D,
    KTrajectoryRpe,
    KTrajectorySpiral2D,
    KTrajectorySunflowerGoldenRpe,
)

from tests.data import PulseqRadialTestSeq


def test_KTrajectoryRadial2D() -> None:
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

    assert trajectory.kz.shape == (1, 1, 1, 1, 1)
    assert trajectory.ky.shape == (1, 1, 1, n_k1, n_k0)
    assert trajectory.kx.shape == (1, 1, 1, n_k1, n_k0)


def test_KTrajectoryRpe() -> None:
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

    assert trajectory.kz.shape == (1, 1, n_k2, n_k1, 1)
    assert trajectory.ky.shape == (1, 1, n_k2, n_k1, 1)
    assert trajectory.kx.shape == (1, 1, 1, 1, n_k0)


def test_KTrajectoryRpe_angle() -> None:
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


def test_KTrajectorySunflowerGoldenRpe() -> None:
    """Test shape returned by KTrajectorySunflowerGoldenRpe"""
    n_k0 = 100
    n_k1 = 20
    n_k2 = 10
    k2_idx = torch.arange(n_k2)[:, None, None]
    k1_idx = torch.arange(n_k1)[:, None]

    trajectory_calculator = KTrajectorySunflowerGoldenRpe()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=n_k0 // 2,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
    )

    assert trajectory.shape == (1, 1, n_k2, n_k1, n_k0)


def test_KTrajectoryCartesian() -> None:
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
    assert trajectory.kz.shape == (1, 1, n_k2, 1, 1)
    assert trajectory.ky.shape == (1, 1, 1, n_k1, 1)
    assert trajectory.kx.shape == (1, 1, 1, 1, n_k0)


def test_KTrajectoryCartesian_bipolar() -> None:
    """Partial fourier and reversed readout"""
    n_k0 = 428
    n_k1 = 3
    n_k2 = 2
    k2_idx = torch.arange(n_k2)[:, None, None]
    k1_idx = torch.arange(n_k1)[:, None]

    reversed_readout_mask = torch.zeros(n_k1, dtype=torch.bool)
    reversed_readout_mask[1] = True

    trajectory_calculator = KTrajectoryCartesian()
    trajectory = trajectory_calculator(
        n_k0=n_k0,
        k0_center=172,
        k1_idx=k1_idx,
        k1_center=n_k1 // 2,
        k2_idx=k2_idx,
        k2_center=n_k2 // 2,
        reversed_readout_mask=reversed_readout_mask,
    )
    assert trajectory.kx[..., 0, 172] == 0  # normal readout
    assert trajectory.kx[..., 0, 0] == -172
    assert trajectory.kx[..., 1, 171] == 0  # reversed readout
    assert trajectory.kx[..., 1, 0] == 171


def test_KTrajectoryIsmrmrdRadial(ismrmrd_rad) -> None:
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
    seq_filename = tmp_path_factory.mktemp('mr2') / 'radial.seq'
    seq = PulseqRadialTestSeq(seq_filename, n_k0=256, n_spokes=10)
    return seq


def test_KTrajectoryPulseq(pulseq_example_rad_seq) -> None:
    """Test pulseq File reader with valid seq File."""
    # TODO: Test with invalid seq file
    trajectory_calculator = KTrajectoryPulseq(seq_path=pulseq_example_rad_seq.seq_filename)
    trajectory = trajectory_calculator(
        n_k0=pulseq_example_rad_seq.n_k0, encoding_matrix=pulseq_example_rad_seq.encoding_matrix
    )

    kx_test = rearrange(pulseq_example_rad_seq.traj_analytical.kx, 'other coils k2 k1 k0 -> (other k2 k1) coils 1 1 k0')
    kx_test = kx_test * pulseq_example_rad_seq.encoding_matrix.x / (2 * kx_test.abs().max())
    ky_test = rearrange(pulseq_example_rad_seq.traj_analytical.ky, 'other coils k2 k1 k0 -> (other k2 k1) coils 1 1 k0')
    ky_test = ky_test * pulseq_example_rad_seq.encoding_matrix.y / (2 * ky_test.abs().max())

    torch.testing.assert_close(trajectory.kx.to(torch.float32), kx_test.to(torch.float32), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(trajectory.ky.to(torch.float32), ky_test.to(torch.float32), atol=1e-2, rtol=1e-3)


def test_KTrajectoryCartesian_random(acceleration: int = 2, n_k: int = 64) -> None:
    """Test the generation of a 2D gaussian variable density pattern"""

    traj = KTrajectoryCartesian.gaussian_variable_density(n_k, acceleration=acceleration, n_other=(2, 3), n_center=8)

    assert traj.kx.shape == (1, 1, 1, 1, 1, n_k)
    assert traj.ky.shape == (2, 3, 1, 1, n_k // acceleration, 1)

    lines1 = traj.ky[0, 0].unique()
    lines2 = traj.ky[0, 1].unique()
    assert not torch.allclose(lines1, lines2)

    assert len(lines1) == n_k // acceleration
    for center_idx in range(-4, 4):
        assert center_idx in lines1


def test_KTrajectoryCartesian_fullysampled() -> None:
    """Test the generation of a fully sampled Cartesian trajectory"""
    traj = KTrajectoryCartesian.fullysampled(SpatialDimension(10, 64, 64))
    assert traj.kx.shape == (1, 1, 1, 1, 64)
    assert traj.ky.shape == (1, 1, 1, 64, 1)
    assert traj.kz.shape == (1, 1, 10, 1, 1)
    assert len(traj.kx.unique()) == 64
    assert len(traj.ky.unique()) == 64
    assert len(traj.kz.unique()) == 10
    assert traj.kx.diff().unique() == 1


@pytest.mark.parametrize('acceleration', [1, 16])
def test_KTrajectoryCartesian_random_edgecases(acceleration: int, n_k=128) -> None:
    """Test the generation of a 2D gaussian variable density pattern"""
    traj = KTrajectoryCartesian.gaussian_variable_density(n_k, acceleration=acceleration, n_other=(2, 3), n_center=8)

    assert traj.kx.shape == (1, 1, 1, 1, 1, n_k)
    assert traj.ky.shape == (1, 1, 1, 1, n_k // acceleration, 1)

    for center_idx in range(-4, 4):
        assert center_idx in traj.ky.ravel()


def test_KTrajectorySpiral() -> None:
    """Test the generation of a 2D spiral trajectory"""
    trajectory_calculator = KTrajectorySpiral2D()
    trajectory = trajectory_calculator(
        n_k0=1024, k1_idx=torch.arange(4)[:, None], encoding_matrix=SpatialDimension(1, 256, 256)
    )
    assert trajectory.kz.shape == (1, 1, 1, 1, 1)
    assert trajectory.ky.shape == (1, 1, 1, 4, 1024)
    assert trajectory.kx.shape == (1, 1, 1, 4, 1024)
