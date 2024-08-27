"""Tests for algorithms to calculate the DCF with voronoi."""

import math

import pytest
import torch
from einops import repeat
from mrpro.algorithms.dcf import dcf_1d, dcf_2d3d_voronoi
from mrpro.data import KTrajectory


def example_traj_rad_2d(n_kr, n_ka, phi0=0.0, broadcast=True):
    """Create 2D radial trajectory with uniform angular gap."""
    krad = repeat(torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr, 'k0 -> other k2 k1 k0', other=1, k2=1, k1=1)
    kang = repeat(
        torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka) + phi0, 'k1 -> other k2 k1 k0', other=1, k2=1, k0=1
    )
    kz = torch.zeros(1, 1, 1, 1)
    ky = torch.sin(kang) * krad
    kx = torch.cos(kang) * krad
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


@pytest.mark.parametrize(
    ('n_kr', 'n_ka', 'phi0', 'broadcast'),
    [
        (100, 20, 0, True),
        (100, 1, 0, True),
        (100, 20, torch.pi / 4, True),
        (100, 1, torch.pi / 4, True),
        (100, 1, 0, False),
    ],
)
def test_dcf_rad_traj_voronoi(n_kr, n_ka, phi0, broadcast):
    """Compare voronoi-based dcf calculation for 2D radial trajectory to
    analytical solution."""
    # 2D radial trajectory
    traj = example_traj_rad_2d(n_kr, n_ka, phi0, broadcast)
    trajectory = traj.as_tensor()

    if n_ka > 1:  # only for for multiple spokes, analytical dcf is defined
        dcf = dcf_2d3d_voronoi(trajectory[1:3, 0, ...])
        krad_idx = torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr)
        dcf_analytical = torch.pi / n_ka * torch.abs(krad_idx) * (1 / n_kr) ** 2
        dcf_analytical[krad_idx == 0] = 2 * torch.pi / n_ka * 1 / 8 * (1 / n_kr) ** 2
        dcf_analytical = torch.repeat_interleave(repeat(dcf_analytical, 'k0->k2 k1 k0', k1=1, k2=1), n_ka, dim=-2)
        # Do not test outer points because they have to be approximated and cannot be calculated
        # accurately using voronoi
        torch.testing.assert_close(dcf_analytical[:, :, 1:-1], dcf[:, :, 1:-1])
    else:
        dcf = dcf_1d(trajectory[0, ...])
        dcf_ptp = dcf.max() - dcf.min()
        assert dcf_ptp / dcf.max() < 0.1, 'DCF for a single spoke should be constant-ish'
        assert dcf.sum() > 1e-3, 'DCF sum should not be zero'
        assert dcf.shape == traj.broadcasted_shape, 'DCF shape should match broadcasted trajectory shape'


@pytest.mark.parametrize(('n_k2', 'n_k1', 'n_k0'), [(40, 16, 20), (1, 2, 2)])
def test_dcf_3d_cart_traj_broadcast_voronoi(n_k2, n_k1, n_k0):
    """Compare voronoi-based dcf calculation for broadcasted 3D regular
    Cartesian trajectory to analytical solution which is 1 for each k-space
    point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    kx = repeat(torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0), 'k0 -> other k2 k1 k0', other=1, k2=1, k1=1)
    ky = repeat(torch.linspace(-n_k1 // 2, n_k1 // 2 - 1, n_k1), 'k1 -> other k2 k1 k0', other=1, k2=1, k0=1)
    kz = repeat(torch.linspace(-n_k2 // 2, n_k2 // 2 - 1, n_k2), 'k2 -> other k2 k1 k0', other=1, k1=1, k0=1)
    trajectory = KTrajectory(kx, ky, kz)

    # Analytical dcf
    dcf_analytical = torch.ones((1, n_k2, n_k1, n_k0))
    # calculate dcf
    dcf = dcf_2d3d_voronoi(trajectory.as_tensor())
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi.
    torch.testing.assert_close(dcf[:, 1:-1, 1:-1, 1:-1], dcf_analytical[:, 1:-1, 1:-1, 1:-1])


@pytest.mark.parametrize(('n_k2', 'n_k1', 'n_k0'), [(40, 16, 20), (1, 2, 2)])
def test_dcf_3d_cart_full_traj_voronoi(n_k2, n_k1, n_k0):
    """Compare voronoi-based dcf calculation for full 3D regular Cartesian
    trajectory to analytical solution which is 1 for each k-space point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    ky, kz, kx = torch.meshgrid(
        torch.linspace(-n_k1 // 2, n_k1 // 2 - 1, n_k1),
        torch.linspace(-n_k2 // 2, n_k2 // 2 - 1, n_k2),
        torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0),
        indexing='xy',
    )
    trajectory = KTrajectory(
        repeat(kz, '... -> other ...', other=1),
        repeat(ky, '... -> other ...', other=1),
        repeat(kx, '... -> other ...', other=1),
        repeat_detection_tolerance=None,
    )
    # Analytical dcf
    dcf_analytical = torch.ones((1, n_k2, n_k1, n_k0))
    dcf = dcf_2d3d_voronoi(trajectory.as_tensor())
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf[:, 1:-1, 1:-1, 1:-1], dcf_analytical[:, 1:-1, 1:-1, 1:-1])


@pytest.mark.parametrize(
    ('n_k2', 'n_k1', 'n_k0', 'k2_steps', 'k1_steps', 'k0_steps'),
    [(30, 20, 10, (1.0, 0.5, 0.25), (1.0, 0.5), (1.0,))],
)
def test_dcf_3d_cart_nonuniform_traj_voronoi(n_k2, n_k1, n_k0, k2_steps, k1_steps, k0_steps):
    """Compare voronoi-based dcf calculation for 3D nonuniform Cartesian
    trajectory to analytical solution which is 1 for each k-space point."""

    def k_range(n: int, *steps: float):
        """Create a tensor with n values, steps apart."""
        r = torch.tensor(steps).repeat(math.ceil(n / len(steps))).ravel()[:n]
        r = torch.cumsum(r, 0)
        r -= r.mean()
        return r

    k0_range = k_range(n_k0, *k0_steps)
    k1_range = k_range(n_k1, *k1_steps)
    k2_range = k_range(n_k2, *k2_steps)

    ky_full, kz_full, kx_full = torch.meshgrid(k1_range, k2_range, k0_range, indexing='xy')
    trajectory_full = KTrajectory(
        repeat(kz_full, '... -> other ...', other=1),
        repeat(ky_full, '... -> other ...', other=1),
        repeat(kx_full, '... -> other ...', other=1),
        repeat_detection_tolerance=None,
    )

    kx_broadcast = repeat(k0_range, 'k0 -> k2 k1 k0', k2=1, k1=1)
    ky_broadcast = repeat(k1_range, 'k1 -> k2 k1 k0', k2=1, k0=1)
    kz_broadcast = repeat(k2_range, 'k2 -> k2 k1 k0', k1=1, k0=1)
    trajectory_broadcast = KTrajectory(
        repeat(kz_broadcast, '... -> other ...', other=1),
        repeat(ky_broadcast, '... -> other ...', other=1),
        repeat(kx_broadcast, '... -> other ...', other=1),
        repeat_detection_tolerance=None,
    )

    # Sanity check inputs
    torch.testing.assert_close(trajectory_full.as_tensor(), trajectory_broadcast.as_tensor())
    assert trajectory_full.broadcasted_shape == (1, len(k2_range), len(k1_range), len(k0_range))
    torch.testing.assert_close(kx_full[0, 0, :], k0_range)  # kx changes along k0
    torch.testing.assert_close(ky_full[0, :, 0], k1_range)  # ky changes along k1
    torch.testing.assert_close(kz_full[:, 0, 0], k2_range)  # kz changes along k2

    dcf_full = dcf_2d3d_voronoi(trajectory_full.as_tensor()[1:3, 0, ...])
    dcf_broadcast = dcf_2d3d_voronoi(trajectory_broadcast.as_tensor()[1:3, 0, ...])

    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf_full[1:-1, 1:-1, 1:-1], dcf_broadcast[1:-1, 1:-1, 1:-1])
