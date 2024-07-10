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

import pytest
import torch
from mrpro.data import DcfData, KTrajectory


def example_traj_rpe(n_kr, n_ka, n_k0, broadcast=True):
    """Create RPE trajectory with uniform angular gap."""
    krad = torch.linspace(-n_kr // 2, n_kr // 2 - 1, n_kr) / n_kr
    kang = torch.linspace(0, n_ka - 1, n_ka) * (torch.pi / n_ka)
    kz = (torch.sin(kang[:, None]) * krad[None, :])[None, :, :, None]
    ky = (torch.cos(kang[:, None]) * krad[None, :])[None, :, :, None]
    kx = (torch.linspace(-n_k0 // 2, n_k0 // 2 - 1, n_k0) / n_k0)[None, None, None, :]
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


@pytest.mark.parametrize(('n_kr', 'n_ka', 'n_k0'), [(10, 6, 20), (10, 1, 20), (10, 6, 1)])
def test_dcf_rpe_traj_voronoi(n_kr, n_ka, n_k0):
    """Voronoi-based dcf calculation for RPE trajectory."""
    trajectory = example_traj_rpe(n_kr, n_ka, n_k0)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == (1, n_ka, n_kr, n_k0)


@pytest.mark.parametrize(('n_kr', 'n_ki', 'n_ka'), [(10, 2, 1)])
def test_dcf_spiral_traj_voronoi(n_kr, n_ki, n_ka):
    """Voronoi-based dcf calculation for spiral trajectory."""
    # nkr points along each spiral arm, nki turns per spiral arm, nka spiral arms
    trajectory = example_traj_spiral_2d(n_kr, n_ki, n_ka)
    dcf = DcfData.from_traj_voronoi(trajectory)
    assert dcf.data.shape == trajectory.broadcasted_shape


def test_dcf_spiral_traj_voronoi_singlespiral():
    """For three z-stacked spirals in the x,y plane, the center spiral should
    be the same as a single 2D spiral.

    Issue #84
    """
    n_kr = 100  # points along each spiral ar
    n_ki = 5  # turns per spiral arm spirals nka spiral arms
    trajectory_single = example_traj_spiral_2d(n_kr, n_ki, 1)

    # A new trajectroy with three spirals stacked in z direction.
    three_spirals: torch.Tensor = trajectory_single.as_tensor().repeat_interleave(repeats=3, dim=-2)
    three_spirals[0, :, :, 0] = -1  # z of first spiral
    three_spirals[0, :, :, 1] = 0  # z of second spiral
    three_spirals[0, :, :, 2] = 1  # z of third spiral
    trajectory_three_dense = KTrajectory.from_tensor(three_spirals, repeat_detection_tolerance=None)
    trajectory_three_broadcast = KTrajectory.from_tensor(three_spirals)

    dcf_single = DcfData.from_traj_voronoi(trajectory_single)
    dcf_three_dense = DcfData.from_traj_voronoi(trajectory_three_dense)
    dcf_three_broadcast = DcfData.from_traj_voronoi(trajectory_three_broadcast)

    ignore_last = int(n_kr / n_ki)  # ignore the outer points of the spirals
    torch.testing.assert_close(dcf_three_dense.data[..., 1, :-ignore_last], dcf_single.data[..., 0, :-ignore_last])
    torch.testing.assert_close(dcf_three_broadcast.data[..., 1, :-ignore_last], dcf_single.data[..., 0, :-ignore_last])


@pytest.mark.cuda()
@pytest.mark.parametrize(('n_kr', 'n_ka', 'n_k0'), [(10, 6, 20)])
def test_dcf_rpe_traj_voronoi_cuda(n_kr, n_ka, n_k0):
    """Voronoi-based dcf calculation for RPE trajectory in CUDA memory."""
    trajectory = example_traj_rpe(n_kr, n_ka, n_k0)
    dcf = DcfData.from_traj_voronoi(trajectory.cuda())
    assert dcf.data.is_cuda
