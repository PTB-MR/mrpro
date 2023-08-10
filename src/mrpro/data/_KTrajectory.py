"""K-space trajectory classes."""

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

from __future__ import annotations

import dataclasses
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from mrpro.data._KHeader import KHeader


class KTrajectory(ABC):
    """Base class for k-space trajectories."""

    @staticmethod
    def _get_shape(header: KHeader) -> tuple[int, ...]:
        """Get the shape of the trajectory for the given header."""
        limits = header.encoding_limits
        other_dim = np.prod(
            [
                getattr(limits, field.name).length
                for field in dataclasses.fields(limits)
                if field.name not in ('k0', 'k1', 'k2', 'segment')
            ]
        )
        shape = (
            other_dim,
            3,
            limits.k2.length,
            limits.k1.length,
            limits.k0.length,
        )
        return shape

    @abstractmethod
    def calc_traj(self, header: KHeader) -> torch.Tensor:
        """Calculate the trajectory for the given header.

        The shape should broadcastable to (prod(all_other_dimensions),
        3, k2, k1, k0)
        """
        ...


class DummyTrajectory(KTrajectory):
    """Dummy trajectory that returns zeros."""

    def calc_traj(self, header: KHeader) -> torch.Tensor:
        """Calculate the trajectory for the given header.

        The shape should broadcastable to (prod(all_other_dimensions),
        3, k2, k1, k0)
        """
        return torch.zeros(self._get_shape(header))


@dataclass
class RadialKTraj2D(KTrajectory):
    """Returns 2D radial trajectory."""

    golden_angle: bool = True
    multi_slice: bool = False

    def calc_traj(self, header: KHeader) -> torch.Tensor:
        """Calculate the 2D radial trajectory for the given header."""

        # # I found that the following in the header file can have some
        # # potential use but the same information can be derived from
        # # self._get_shape(header)
        # spoke_counter = header.acq_info.idx.k1
        # sample_counter = header.acq_info.number_of_samples

        n_spokes, n_readout = self._get_shape(header)[-2], self._get_shape(header)[-1]

        # length of each spoke;
        spoke_length = n_readout

        # golden angle radial increment
        if self.golden_angle is True:
            del_theta = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))

        # create radial coordinates
        ktheta = torch.arange(0, del_theta*n_spokes, del_theta)
        kradius = torch.linspace(-np.pi, np.pi, spoke_length)  # format of trajectories kbNUFFT package expects to be

        # construct transformation matrices
        ktraj = torch.zeros(self._get_shape(header))
        ktraj_init = torch.stack((kradius, torch.zeros(n_readout), torch.zeros(n_readout)))
        for i in range(n_spokes):
            theta = ktheta[i]
            rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                    [torch.sin(theta), torch.cos(theta), 0],
                                    [0, 0, 1]])
            ktraj[..., i, :] = torch.matmul(rot_mat, ktraj_init).unsqueeze(1).unsqueeze(0)

        # skip if all optional trajectory information are missing else do something
        if torch.all(header.acq_info.trajectory_dimensions == 0):
            pass
        else:
            pass  # do something: replace the calculated trajectory information with the already present one

        return ktraj
