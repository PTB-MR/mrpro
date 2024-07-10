"""Tests for algoritms to calculate the DCF with voronoi."""
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

#%%
import math

import pytest
import torch
from mrpro.data import KTrajectory
from mrpro.algorithms.dcf import dcf_1d, dcf_2d3d_voronoi
from mrpro.utils import smap
#%%

def example_traj_rpe(n_kr, n_ka, n_k0, broadcast=True):
    """Create RPE trajectory with uniform angular gap."""
    krad = torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr
    kang = torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka)
    kz = (torch.sin(kang[:, None]) * krad[None, :])[None, :, :, None]
    ky = (torch.cos(kang[:, None]) * krad[None, :])[None, :, :, None]
    kx = (torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0) / n_k0)[None, None, None, :]
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


def example_traj_rad_2d(n_kr, n_ka, phi0=0.0, broadcast=True):
    """Create 2D radial trajectory with uniform angular gap."""
    krad = torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr
    kang = torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka) + phi0
    kz = torch.zeros(1, 1, 1, 1)
    ky = (torch.sin(kang[:, None]) * krad[None, :])[None, None, :, :]
    kx = (torch.cos(kang[:, None]) * krad[None, :])[None, None, :, :]
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory


def example_traj_spiral_2d(n_kr, n_ki, n_ka, broadcast=True) -> KTrajectory:
    """Create 2D spiral trajectory with n_kr points along each spiral arm, n_ki
    turns per spiral arm and n_ka spiral arms."""
    ang = torch.linspace(0, 2 * torch.pi * n_ki, n_kr)
    start_ang = torch.linspace(0, 2 * torch.pi * (1 - 1 / n_ka), n_ka)
    kz = torch.zeros(1, 1, 1, 1)
    kx = (ang[None, :] * torch.cos(ang[None, :] + start_ang[:, None]))[None, None, :, :]
    ky = (ang[None, :] * torch.sin(ang[None, :] + start_ang[:, None]))[None, None, :, :]
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=1e-8 if broadcast else None)
    return trajectory

traj = example_traj_rad_2d(100, 1, 0, True)

#%%

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
        dcf = dcf_2d3d_voronoi(trajectory[1:3,0,...])
        krad_idx = torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr)
        dcf_analytical = torch.pi / n_ka * torch.abs(krad_idx) * (1 / n_kr) ** 2
        dcf_analytical[krad_idx == 0] = 2 * torch.pi / n_ka * 1 / 8 * (1 / n_kr) ** 2
        dcf_analytical = torch.repeat_interleave(dcf_analytical[None, ...], n_ka, dim=0)[None, :, :]
        # Do not test outer points because they have to be approximated and cannot be calculated
        # accurately using voronoi
        torch.testing.assert_close(dcf_analytical[:, :, 1:-1], dcf[:, :, 1:-1])
    else:
        dcf = dcf_1d(trajectory[1:3,0,...])
        dcf_ptp = dcf.max() - dcf.min()
        assert dcf_ptp / dcf.max() < 0.1, 'DCF for a single spoke should be constant-ish'
        assert dcf.sum() > 1e-3, 'DCF sum should not be zero'
        assert dcf.shape == trajectory.broadcasted_shape, 'DCF shape should match broadcasted trajectory shape'


@pytest.mark.parametrize(('n_k2', 'n_k1', 'n_k0'), [(40, 16, 20), (1, 2, 2)])
def test_dcf_3d_cart_traj_broadcast_voronoi(n_k2, n_k1, n_k0):
    """Compare voronoi-based dcf calculation for broadcasted 3D regular
    Cartesian trajectory to analytical solution which is 1 for each k-space
    point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    kx = torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0)[None, None, None, :]
    ky = torch.linspace(-n_k1 // 2, n_k1 // 2 - 1, n_k1)[None, None, :, None]
    kz = torch.linspace(-n_k2 // 2, n_k2 // 2 - 1, n_k2)[None, :, None, None]
    traj = KTrajectory(kx, ky, kz)
    trajectory = traj.as_tensor()

    # Analytical dcf
    dcf_analytical = torch.ones((n_k2, n_k1, n_k0))
    # calculate dcf
    dcf = dcf_2d3d_voronoi(trajectory[1:3,0,...])
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi.
    torch.testing.assert_close(dcf[1:-1, 1:-1, 1:-1], dcf_analytical[1:-1, 1:-1, 1:-1])


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
    trajectory = KTrajectory(kz[None, ...], ky[None, ...], kx[None, ...], repeat_detection_tolerance=None)
    trajectory = trajectory.as_tensor()
    # Analytical dcf
    dcf_analytical = torch.ones((n_k2, n_k1, n_k0))
    dcf = dcf_2d3d_voronoi(trajectory[1:3,0,...])
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf[1:-1, 1:-1, 1:-1], dcf_analytical[1:-1, 1:-1, 1:-1])


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
        kz_full[None, ...],
        ky_full[None, ...],
        kx_full[None, ...],
        repeat_detection_tolerance=None,
    )

    kx_broadcast = k0_range[None, None, :]
    ky_broadcast = k1_range[None, :, None]
    kz_broadcast = k2_range[:, None, None]
    trajectory_broadcast = KTrajectory(
        kz_broadcast[None, ...],
        ky_broadcast[None, ...],
        kx_broadcast[None, ...],
        repeat_detection_tolerance=None,
    )

    # Sanity check inputs
    torch.testing.assert_close(trajectory_full.as_tensor(), trajectory_broadcast.as_tensor())
    assert trajectory_full.broadcasted_shape == (1, len(k2_range), len(k1_range), len(k0_range))
    torch.testing.assert_close(kx_full[0, 0, :], k0_range)  # kx changes along k0
    torch.testing.assert_close(ky_full[0, :, 0], k1_range)  # ky changes along k1
    torch.testing.assert_close(kz_full[:, 0, 0], k2_range)  # kz changes along k2

    dcf_full = dcf_2d3d_voronoi(trajectory_full.as_tensor()[1:3,0,...])
    dcf_broadcast = dcf_2d3d_voronoi(trajectory_broadcast.as_tensor()[1:3,0,...])

    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf_full[1:-1, 1:-1, 1:-1], dcf_broadcast[1:-1, 1:-1, 1:-1])
