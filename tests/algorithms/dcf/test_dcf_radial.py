"""Tests for algorithms to calculate the DCF with voronoi."""

import pytest
import torch
from einops import repeat
from mrpro.algorithms.dcf import dcf_2dradial
from mrpro.data import KTrajectory


def example_traj_rad_2d(n_kr, n_ka, phi0=0.0, broadcast=True):
    """Create 2D radial trajectory with uniform angular gap."""
    krad = repeat(
        torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr,
        'k0 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k1=1,
    )
    kang = repeat(
        torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka) + phi0,
        'k1 -> other coils k2 k1 k0',
        other=1,
        coils=1,
        k2=1,
        k0=1,
    )
    kz = torch.zeros(1, 1, 1, 1, 1)
    ky = torch.sin(kang) * krad
    kx = torch.cos(kang) * krad
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


@pytest.mark.parametrize(
    ('n_kr', 'n_ka', 'phi0', 'broadcast'),
    [
        (20, 20, 0, True),
        (20, 2, 0, True),
        (20, 20, torch.pi / 4, True),
        (20, 2, torch.pi / 4, True),
        (20, 2, 0, False),
    ],
)
def test_dcf_2drad_analytical_equidist(n_kr, n_ka, phi0, broadcast):
    """Compare 2d dcf calculation for 2D radial trajectory to
    analytical solution for 2D equidistant dcf."""
    # 2D radial trajectory
    traj = example_traj_rad_2d(n_kr, n_ka, phi0, broadcast)
    trajectory = traj.as_tensor()

    dcf_2drad = dcf_2dradial(trajectory[1:3, 0, 0, ...])
    krad_idx = torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr)
    dcf_analytical_equidist = torch.pi / n_ka * torch.abs(krad_idx) * (1 / n_kr) ** 2
    dcf_analytical_equidist[krad_idx == 0] = 2 * torch.pi / n_ka * 1 / 8 * (1 / n_kr) ** 2
    dcf_analytical_equidist = torch.repeat_interleave(
        repeat(dcf_analytical_equidist, 'k0->k2 k1 k0', k1=1, k2=1), n_ka, dim=-2
    ).unsqueeze(0)
    # Do not test outer points because they have to be approximated and cannot be calculated exactly
    torch.testing.assert_close(dcf_analytical_equidist[:, :, :, 1:-1], dcf_2drad[:, :, :, 1:-1])
