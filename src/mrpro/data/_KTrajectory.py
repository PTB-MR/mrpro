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

import numpy as np
import torch


@dataclass(slots=True, init=False)
class KTrajectory:
    """K-space trajectory.

    Order of directions is always kz, ky, kx
    Shape of each of kx,ky,kz is (other,k2,k1,k0)

    Example for 2D-Cartesian Trajectories:
        kx changes along k0 and is Frequency Encoding
        ky changes along k2 and is Phase Encoding
        kz is zero(1,1,1,1)
    """

    kz: torch.Tensor  # (other,k2,k1,k0) #phase encoding direction, k2 if Cartesian
    ky: torch.Tensor  # (other,k2,k1,k0) #phase encoding direction, k1 if Cartesian
    kx: torch.Tensor  # (other,k2,k1,k0) #frequency encoding direction, k0 if Cartesian

    @property
    def broadcasted_shape(self) -> tuple[int, ...]:
        """The broadcasted shape of the trajectory."""
        shape = np.broadcast_shapes(self.kx.shape, self.ky.shape, self.kz.shape)
        return tuple(shape)

    def as_tensor(self, stack_dim=0):
        """Tensor representation of the trajectory.

        Parameters
        ----------
        stack_dim:
            The dimension to stack the tensor along.
        """
        shape = self.broadcasted_shape
        return torch.stack([traj.expand(*shape) for traj in (self.kz, self.ky, self.kx)], dim=stack_dim)

    def __init__(
        self, kz: torch.Tensor, ky: torch.Tensor, kx: torch.Tensor, repeat_detection_tolerance: float | None = 1e-8
    ):
        """K-Space Trajectory dataclass.

        Reduces repeated dimensions to singletons if repeat_detection_tolerance
        is not set to None.

        Parameters
        ----------
        kz, ky, kx:
            Trajectory coordinates to set
        repeat_detection_tolerance:
            Tolerance for repeat detection. Set to None to disable.
        """
        if repeat_detection_tolerance is not None:
            kz, ky, kx = (KTrajectory._remove_repeat(tensor, repeat_detection_tolerance) for tensor in (kz, ky, kx))

        self.kz = kz
        self.ky = ky
        self.kx = kx

        try:
            shape = self.broadcasted_shape
        except ValueError:
            raise ValueError('The k-space trajectory dimensions must be broadcastable.')
        if len(shape) != 4:
            raise ValueError('The k-space trajectory tensors should each have 4 dimensions.')

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, stack_dim: int = 0, repeat_detection_tolerance: float | None = 1e-8
    ) -> KTrajectory:
        """Create a KTrajectory from a tensor representation of the trajectory.

        Reduces repeated dimensions to singletons if repeat_detection_tolerance
        is not set to None.


        Parameters
        ----------
        tensor:
            The tensor representation of the trajectory.
            This should be a 5-dim tensor, with (kz,ky,kx) stacked in this order along stack_dim
        stack_dim:
            The dimension in the tensor the directions have been stacked along.
        repeat_detection_tolerance:
            detects if broadcasting can be used, i.e. if dimensions are repeated.
            Set to None to disable.
        """

        kz, ky, kx = torch.unbind(tensor, dim=stack_dim)
        return cls(kz, ky, kx, repeat_detection_tolerance=repeat_detection_tolerance)

    @staticmethod
    def _remove_repeat(tensor: torch.Tensor, tol: float) -> torch.Tensor:
        """Replace dimensions with all equal values with singletons.

        Parameters
        ----------
        tensor:
            The tensor. Must be real
        tol:
            The tolerance
        """
        # TODO: Move to utilities

        def can_be_singleton(dim: int) -> bool:
            # If the distance between min and max is smaller than the tolerance, all values are the same.
            return bool(torch.all((tensor.amax(dim=dim) - tensor.amin(dim=dim)) <= tol).item())

        take_first = slice(0, 1)
        take_all = slice(None)
        index = tuple(take_first if can_be_singleton(dim) else take_all for dim in range(tensor.ndim))
        return tensor[index]
