"""Tests for DcfData class."""

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

import math

import torch

from mrpro.data import DcfData
from mrpro.data import KTrajectory


def example_traj_rpe(nkr, nka, nk0):
    """Create RPE trajectory with uniform angular gap."""
    krad = torch.linspace(-nkr // 2, nkr // 2 - 1, nkr) / nkr
    kang = torch.linspace(0, nka - 1, nka) * (torch.pi / nka)
    kz = (torch.sin(kang[:, None]) * krad[None, :])[None, :, :, None]
    ky = (torch.cos(kang[:, None]) * krad[None, :])[None, :, :, None]
    kx = (torch.linspace(-nk0 // 2, nk0 // 2 - 1, nk0) / nk0)[None, None, None, :]
    ktraj = KTrajectory(kz, ky, kx)
    return ktraj


def example_traj_rad_2d(nkr, nka):
    """Create 2D radial trajectory with uniform angular gap."""
    krad = torch.linspace(-nkr // 2, nkr // 2 - 1, nkr) / nkr
    kang = torch.linspace(0, nka - 1, nka) * (torch.pi / nka)
    kz = torch.zeros(1, 1, 1, 1)
    ky = (torch.sin(kang[:, None]) * krad[None, :])[None, None, :, :]
    kx = (torch.cos(kang[:, None]) * krad[None, :])[None, None, :, :]
    ktraj = KTrajectory(kz, ky, kx)
    return ktraj


def test_dcf_2d_rad_traj_voronoi():
    """Compare voronoi-based dcf calculation for 2D radial trajectory to
    analytical solution."""
    # 2D radial trajectory
    nkr = 100
    nka = 20
    ktraj = example_traj_rad_2d(nkr, nka)

    # Analytical dcf
    krad_idx = torch.linspace(-nkr // 2, nkr // 2 - 1, nkr)
    dcf_analytical = torch.pi / nka * torch.abs(krad_idx) * (1 / nkr) ** 2
    dcf_analytical[krad_idx == 0] = 2 * torch.pi / nka * 1 / 8 * (1 / nkr) ** 2
    dcf_analytical = torch.repeat_interleave(dcf_analytical[None, ...], nka, dim=0)[None, None, :, :]

    dcf = DcfData.from_traj_voronoi(ktraj)

    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf_analytical[:, :, :, 1:-1], dcf.data[:, :, :, 1:-1])


def test_dcf_3d_cart_traj_broadcast_voronoi():
    """Compare voronoi-based dcf calculation for broadcasted 3D regular
    Cartesian trajectory to analytical solution which is 1 for each k-space
    point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    nk0 = 20
    nk1 = 16
    nk2 = 40
    kx = torch.linspace(-nk0 // 2, nk0 // 2 - 1, nk0)[None, None, None, :]
    ky = torch.linspace(-nk1 // 2, nk1 // 2 - 1, nk1)[None, None, :, None]
    kz = torch.linspace(-nk2 // 2, nk2 // 2 - 1, nk2)[None, :, None, None]
    ktraj = KTrajectory(kx, ky, kz)

    # Analytical dcf
    dcf_analytical = torch.ones((1, nk2, nk1, nk0))
    dcf = DcfData.from_traj_voronoi(ktraj)
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf.data[:, 1:-1, 1:-1, 1:-1], dcf_analytical[:, 1:-1, 1:-1, 1:-1])


def test_dcf_3d_cart_full_traj_voronoi():
    """Compare voronoi-based dcf calculation for full 3D regular Cartesian
    trajectory to analytical solution which is 1 for each k-space point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    nk0 = 20
    nk1 = 16
    nk2 = 40
    ky, kz, kx = torch.meshgrid(
        torch.linspace(-nk1 // 2, nk1 // 2 - 1, nk1),
        torch.linspace(-nk2 // 2, nk2 // 2 - 1, nk2),
        torch.linspace(-nk0 // 2, nk0 // 2 - 1, nk0),
        indexing='xy',
    )
    ktraj = KTrajectory(kz[None, ...], ky[None, ...], kx[None, ...], repeat_detection_tolerance=None)

    # Analytical dcf
    dcf_analytical = torch.ones((1, nk2, nk1, nk0))
    dcf = DcfData.from_traj_voronoi(ktraj)
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf.data[:, 1:-1, 1:-1, 1:-1], dcf_analytical[:, 1:-1, 1:-1, 1:-1])


def test_dcf_3d_cart_nonuniform_traj_voronoi():
    """Compare voronoi-based dcf calculation for 3D nonunifrm Cartesian
    trajectory to analytical solution which is 1 for each k-space point."""

    def k_range(N: int, *steps: float):
        """Create a tensor with N values, steps appart."""
        r = torch.tensor(steps).repeat(math.ceil(N / len(steps))).ravel()[:N]
        r = torch.cumsum(r, 0)
        r -= r.mean()
        return r

    k0_range = k_range(10, 1.0, 0.5)
    k1_range = k_range(20, 1.0, 0.5, 0.25)
    k2_range = k_range(30, 1.0)

    ky_full, kz_full, kx_full = torch.meshgrid(
        k1_range,
        k2_range,
        k0_range,
        indexing='xy',
    )
    ktraj_full = KTrajectory(
        kz_full[None, ...], ky_full[None, ...], kx_full[None, ...], repeat_detection_tolerance=None
    )

    kx_broadcast = k0_range[None, None, :]
    ky_broadcast = k1_range[None, :, None]
    kz_broadcast = k2_range[:, None, None]
    ktraj_broadcast = KTrajectory(
        kz_broadcast[None, ...], ky_broadcast[None, ...], kx_broadcast[None, ...], repeat_detection_tolerance=None
    )

    # Sanity check inputs
    torch.testing.assert_close(ktraj_full.as_tensor(), ktraj_broadcast.as_tensor())
    assert ktraj_full.broadcasted_shape == (1, len(k2_range), len(k1_range), len(k0_range))
    torch.testing.assert_close(kx_full[0, 0, :], k0_range)  # kx changes along k0
    torch.testing.assert_close(ky_full[0, :, 0], k1_range)  # ky changes along k1
    torch.testing.assert_close(kz_full[:, 0, 0], k2_range)  # kz changes along k2

    dcf_full = DcfData.from_traj_voronoi(ktraj_full)
    dcf_broadcast = DcfData.from_traj_voronoi(ktraj_broadcast)

    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf_full.data[:, 1:-1, 1:-1, 1:-1], dcf_broadcast.data[:, 1:-1, 1:-1, 1:-1])


def test_dcf_rpe_traj_voronoi():
    """Voronoi-based dcf calculation for RPE trajectory."""
    # RPE trajectory
    nkr = 10
    nka = 6
    nk0 = 20
    ktraj = example_traj_rpe(nkr, nka, nk0)
    dcf = DcfData.from_traj_voronoi(ktraj)
    assert dcf.data.shape == (1, nka, nkr, nk0)
