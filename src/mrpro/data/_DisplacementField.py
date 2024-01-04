"""Displacement field dataclass."""

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


@dataclass(slots=True, init=False)
class DisplacementField:
    """Displacement field.

    A displacement field describes the position change of each voxel
    (i.e. dx for a transformation x -> x + dx). Dx has to be in voxel
    dimensions.

    Order of directions is always z, y, x Shape of each of fx,fy,fz is
    (motion_state, other, z, y, x)
    """

    fz: torch.Tensor  # (motion_state, other, z, y, x)
    fy: torch.Tensor  # (motion_state, other, z, y, x)
    fx: torch.Tensor  # (motion_state, other, z, y, x)

    def as_tensor(self, stack_dim=0):
        """Tensor representation of the displacement fields.

        Parameters
        ----------
        stack_dim
            The dimension to stack the tensor along.
        """
        return torch.stack((self.fz, self.fy, self.fx), dim=stack_dim)

    def __init__(self, fz: torch.Tensor, fy: torch.Tensor, fx: torch.Tensor) -> None:
        """Displacement field dataclass.

        Parameters
        ----------
        fz, fy, fx
            displacement fields to set
        """
        if fz.shape != fy.shape or fz.shape != fx.shape:
            raise ValueError('fz, fy and fx must have the same dimensions.')
        if len(fz.shape) != 5:
            raise ValueError('The displacement field tensors should each have 5 dimensions.')

        self.fz = fz
        self.fy = fy
        self.fx = fx

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, stack_dim: int = 0) -> DisplacementField:
        """Create a DisplacementField from a tensor.

        Parameters
        ----------
        tensor
            The tensor representation of the displacement fields.
            This should be a 6-dim tensor, with (fz,fy,fx) stacked in this order along stack_dim
        stack_dim
            The dimension in the tensor the directions have been stacked along.
        """

        fz, fy, fx = torch.unbind(tensor, dim=stack_dim)
        return cls(fz, fy, fx)
