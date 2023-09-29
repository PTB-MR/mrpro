"""K-space trajectory base class."""

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

from abc import ABC
from abc import abstractmethod

import torch

from mrpro.data._KHeader import KHeader
from mrpro.data._KTrajectory import KTrajectory


class KTrajectoryCalculator(ABC):
    """Base class for k-space trajectories."""

    @abstractmethod
    def __call__(self, header: KHeader) -> KTrajectory:
        """Calculate the trajectory for given KHeader.

        The shapes of kz, ky and kx of the calculated trajectory must be
        broadcastable to (prod(all_other_dimensions), k2, k1, k0).
        """
        ...


class DummyTrajectory(KTrajectoryCalculator):
    """Simple Dummy trajectory that returns zeros.

    Shape will fit to all data. Only used as dummy for testing.
    """

    def __call__(self, header: KHeader) -> KTrajectory:
        """Calculate dummy trajectory for given KHeader."""
        kx = torch.zeros(1, 1, 1, header.encoding_limits.k0.length)
        ky = kz = torch.zeros(1, 1, 1, 1)
        return KTrajectory(kz, ky, kx)
