"""Tests for DcfData class."""

import pytest
import torch
from einops import repeat
from mrpro.data import DcfData, KTrajectory
from mrpro.utils import RandomGenerator


def example_traj_rpe(n_ka: int, n_kr: int, n_k0: int, broadcast: bool = True):
    """Create RPE trajectory with uniform angular gap."""
    krad = repeat(
        torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr,
        'k1 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k0=1,
    )
    kang = repeat(
        torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka),
        'k2 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k1=1,
        k0=1,
    )
    kz = torch.sin(kang) * krad
    ky = torch.cos(kang) * krad
    kx = repeat(
        torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0) / n_k0,
        'k0 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k1=1,
    )
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


def example_traj_spiral_2d(n_kr: int, n_ki: int, n_ka: int, broadcast: bool = True) -> KTrajectory:
    """Create 2D spiral trajectory with n_kr points along each spiral arm, n_ki
    turns per spiral arm and n_ka spiral arms."""
    ang = repeat(
        torch.linspace(0, 2 * torch.pi * n_ki, n_kr), 'k0 -> other coils k2 k1 k0', other=1, coils=1, k2=1, k1=1
    )
    start_ang = repeat(
        torch.linspace(0, 2 * torch.pi * (1 - 1 / n_ka), n_ka),
        'k1 -> other  coils  k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k0=1,
    )
    kz = torch.zeros(1, 1, 1, 1, 1)
    kx = ang * torch.cos(ang + start_ang)
    ky = ang * torch.sin(ang + start_ang)
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


@pytest.mark.parametrize(
    ('n_ka', 'n_kr', 'n_k0', 'expected_shape'),
    [
        (6, 10, 20, (1, 1, 1, 10, 1)),  # no k2 or k0, dcf are the same and reduced
        (1, 10, 20, (1, 1, 1, 1, 1)),  # without angular dimension all dcf are the same
        (6, 10, 1, (1, 1, 1, 10, 1)),
    ],
)
def test_dcf_rpe_traj_voronoi(n_ka: int, n_kr: int, n_k0: int, expected_shape: tuple[int, ...]) -> None:
    """Voronoi-based dcf calculation for RPE trajectory."""
    trajectory = example_traj_rpe(n_ka, n_kr, n_k0)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == expected_shape


@pytest.mark.parametrize(('n_kr', 'n_ki', 'n_ka'), [(10, 2, 1)])
def test_dcf_spiral_traj_voronoi(n_kr: int, n_ki: int, n_ka: int) -> None:
    """Voronoi-based dcf calculation for spiral trajectory."""
    # nkr points along each spiral arm, nki turns per spiral arm, nka spiral arms
    trajectory = example_traj_spiral_2d(n_kr, n_ki, n_ka)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == trajectory.shape


def test_dcf_spiral_traj_voronoi_singlespiral() -> None:
    """For three z-stacked spirals in the x,y plane, the center spiral should
    be the same as a single 2D spiral.
    """
    n_kr = 30  # points along each spiral arm
    n_ki = 5  # turns per spiral arm
    trajectory_single = example_traj_spiral_2d(n_kr, n_ki, 1)

    # A new trajectroy with three spirals stacked in z direction.
    three_spirals: torch.Tensor = trajectory_single.as_tensor().repeat_interleave(repeats=3, dim=-2)
    three_spirals[0, :, :, :, 0] = -1  # z of first spiral
    three_spirals[0, :, :, :, 1] = 0  # z of second spiral
    three_spirals[0, :, :, :, 2] = 1  # z of third spiral
    trajectory_three = KTrajectory.from_tensor(three_spirals)

    dcf_single = DcfData.from_traj_voronoi(trajectory_single)
    dcf_three = DcfData.from_traj_voronoi(trajectory_three)

    ignore_last = int(n_kr / n_ki)  # ignore the outer points of the spirals
    torch.testing.assert_close(dcf_three.data[..., :-ignore_last], dcf_single.data[..., :-ignore_last])


@pytest.mark.cuda
@pytest.mark.parametrize(('n_ka', 'n_kr', 'n_k0'), [(10, 6, 20)])
def test_dcf_rpe_traj_voronoi_cuda(n_ka: int, n_kr: int, n_k0: int) -> None:
    """Voronoi-based dcf calculation for RPE trajectory in CUDA memory."""
    trajectory = example_traj_rpe(n_ka, n_kr, n_k0)
    dcf = DcfData.from_traj_voronoi(trajectory.cuda())
    assert dcf.data.is_cuda


def test_dcf_broadcast() -> None:
    """Test broadcasting within voronoi dcf calculation."""
    rng = RandomGenerator(0)
    # kx and ky force voronoi calculation and need to be broadcasted
    kx = rng.float32_tensor((1, 1, 1, 4, 4))
    ky = rng.float32_tensor((1, 1, 4, 1, 4))
    kz = torch.zeros(1, 1, 1, 1, 1)
    trajectory = KTrajectory(kz, ky, kx)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == trajectory.shape
