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

import torch
from einops import rearrange

from mrpro.data import DcfData


def example_traj_rad_2d(nkr, nka):
    """Create 2D radial trajectory with uniform angular gap."""
    krad = torch.linspace(-nkr // 2, nkr // 2 - 1, nkr) / nkr
    kang = torch.linspace(0, nka - 1, nka) * (torch.pi / nka)
    kx = torch.matmul(torch.cos(kang[:, None]), krad[None, :])[None, None, None, :, :]
    ky = torch.matmul(torch.sin(kang[:, None]), krad[None, :])[None, None, None, :, :]
    ktraj = torch.concatenate((kx, ky), dim=1)
    return ktraj


def example_traj_rpe(nkr, nka, nk0):
    """Create RPE trajectory with uniform angular gap."""
    ktraj = example_traj_rad_2d(nkr, nka)
    ktraj = torch.repeat_interleave(ktraj[:, :, 0, :, :, None], nk0, dim=4)
    k0 = torch.linspace(-nk0 // 2, nk0 // 2 - 1, nk0) / nk0
    k0 = torch.tile(k0[None, None, None, None, :], dims=[1, 1, nka, nkr, 1])
    return torch.cat((k0, ktraj), dim=1)


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
    dcf_analytical = torch.repeat_interleave(dcf_analytical[None, ...], nka, dim=0)[None, None, None, :, :]

    dcf = DcfData.from_traj_voronoi(ktraj)

    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf_analytical[:, :, :, :, 1:-1], dcf.data[:, :, :, :, 1:-1])


def test_dcf_3d_cart_traj_voronoi():
    """Compare voronoi-based dcf calculation for 3D regular Cartesian
    trajectory to analytical solution which is 1 for each k-space point."""
    # 3D trajectory with points on Cartesian grid with step size of 1
    nk0 = 20
    nk1 = 16
    nk2 = 40
    k0, k1, k2 = torch.meshgrid(
        torch.linspace(-nk1 // 2, nk1 // 2 - 1, nk1),
        torch.linspace(-nk2 // 2, nk2 // 2 - 1, nk2),
        torch.linspace(-nk0 // 2, nk0 // 2 - 1, nk0),
        indexing='xy',
    )
    ktraj = rearrange([k0, k1, k2], '(dir other) k2 k1 k0 -> other dir k2 k1 k0', other=1)

    # Analytical dcf
    dcf_analytical = torch.ones((1, 1, nk2, nk1, nk0))

    dcf = DcfData.from_traj_voronoi(ktraj)
    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    torch.testing.assert_close(dcf.data[:, :, 1:-1, 1:-1, 1:-1], dcf_analytical[:, :, 1:-1, 1:-1, 1:-1])


def test_dcf_rpe_traj_voronoi():
    """Voronoi-based dcf calculation for RPE trajectory."""
    # RPE trajectory
    nkr = 10
    nka = 6
    nk0 = 20
    ktraj = example_traj_rpe(nkr, nka, nk0)

    dcf_rpe = DcfData.from_rpe_traj_voronoi(ktraj)
    dcf_3d = DcfData.from_traj_voronoi(ktraj)

    # Do not test outer points because they have to be approximated and cannot be calculated
    # accurately using voronoi
    dcf_rpe.data = dcf_rpe.data[:, :, :, 1:-1, 1:-1]
    dcf_3d.data = dcf_3d.data[:, :, :, 1:-1, 1:-1]

    # Compare normalized values
    torch.testing.assert_close(dcf_rpe.data / torch.sum(dcf_rpe.data), dcf_3d.data / torch.sum(dcf_3d.data))
