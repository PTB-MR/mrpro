"""KTrajectory dataclass."""

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
from dataclasses import dataclass
import torch
import numpy as np


@dataclass(slots=True)
class KTrajectory:
    """k-space trajectory"""

    kx: torch.Tensor
    ky: torch.Tensor
    kz: torch.Tensor

    @property
    def broadcasted_shape(self) -> tuple[int, ...]:
        """The broadcasted shape of the trajectory."""
        shape = np.broadcast_shapes(self.kx.shape, self.ky.shape, self.kz.shape)
        return tuple(shape)

    def as_tensor(self, stack_dim=0):
        """A tensor representation of the trajectory.

        Parameters
        ----------
        stack_dim:
            The dimension to stack the tensor along.
        """
        shape = self.broadcasted_shape
        return torch.stack([traj.expand(*shape) for traj in (self.kx, self.ky, self.kz)], dim=stack_dim)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, stack_dim=0) -> KTrajectory:
        """Create a KTrajectory from a tensor representation of the trajectory.

        Parameters
        ----------
        tensor:
            The tensor representation of the trajectory.
        stack_dim:
            The dimension in the tensor the directions have been stacked along.
        """
        return cls(*torch.split(tensor, 1, dim=stack_dim))

    def __post_init__(self):
        try:
            shape = self.broadcasted_shape
            if len(shape) != 4:
                raise ValueError("The k-space trajectory tensors should each have 4 dimensions.")
        except ValueError:
            raise ValueError("The k-space trajectory dimensions must be broadcastable.")
