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
                if field.name
                not in ('kspace_encoding_step_0', 'kspace_encoding_step_1', 'kspace_encoding_step_2', 'segment')
            ]
        )
        shape = (
            other_dim,
            3,
            limits.kspace_encoding_step_2.length,
            limits.kspace_encoding_step_1.length,
            limits.kspace_encoding_step_0.length,
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
