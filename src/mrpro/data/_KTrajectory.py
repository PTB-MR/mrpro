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

    @abstractmethod
    def calc_traj(self, header: KHeader) -> torch.Tensor:
        """Calculate the trajectory for the given header.

        The shape should broadcastable to (prod(all_other_dimensions),
        3, k2, k1, k0)
        """
        ...


class DummyTrajectory(KTrajectory):
    """
    Simple Dummy trajectory that returns zeros.
    Shape will not fit to all data.
    Only used until we implement proper trajectories
    """

    @staticmethod
    def _get_shape(header: KHeader) -> tuple[int, ...]:
        """
        Get the shape of a basic dummy trajectory for the given header.
        Assumes fully sampled data. Do not use outside of testing.
        """
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


    def calc_traj(self, header: KHeader) -> torch.Tensor:
        """Calculate the trajectory for the given header.

        The shape should broadcastable to (prod(all_other_dimensions),
        3, k2, k1, k0)
        """
        return torch.zeros(self._get_shape(header))
