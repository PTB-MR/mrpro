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

from mrpro.data.enums import TrajType
from mrpro.utils import remove_repeat


@dataclass(slots=True, init=False, frozen=True)
class KTrajectory:
    """K-space trajectory.

    Order of directions is always kz, ky, kx
    Shape of each of kx,ky,kz is (other,k2,k1,k0)

    Example for 2D-Cartesian Trajectories:
        kx changes along k0 and is Frequency Encoding
        ky changes along k2 and is Phase Encoding
        kz is zero(1,1,1,1)
    """

    kz: torch.Tensor
    """(other,k2,k1,k0), phase encoding direction k2 if Cartesian."""

    ky: torch.Tensor
    """(other,k2,k1,k0), phase encoding direction k1 if Cartesian."""

    kx: torch.Tensor
    """(other,k2,k1,k0), frequency encoding direction k0 if Cartesian."""

    _type_along_kzyx: tuple[TrajType, TrajType, TrajType]
    _type_along_k210: tuple[TrajType, TrajType, TrajType]
    _expected_tensor_versions: tuple[int, int, int]

    # Type of trajectory
    _type_kz_ky_kx: list[TrajType]  # for kz, ky and kx directions
    _type_k2_k1_k0: list[TrajType]  # for k2, k1 and k0 directions

    # Tolerance for trajectory type estimation (i.e. how close do the values have to be to grid points)
    type_tolerance: float

    # Tensor version of kz, ky and kx to detect any changes which require type update
    _traj_version: list[int]

    @property
    def broadcasted_shape(self) -> tuple[int, ...]:
        """The broadcasted shape of the trajectory.

        Returns
        -------
            broadcasted shape of trajectory
        """
        shape = np.broadcast_shapes(self.kx.shape, self.ky.shape, self.kz.shape)
        return tuple(shape)

    def _traj_types(self, tolerance) -> tuple[tuple[TrajType, TrajType, TrajType], tuple[TrajType, TrajType, TrajType]]:
        """Calculate the trajectory type along kzkykx and k2k1k0.

        Checks if the entries of the trajectory along certain dimensions
            - are of shape 1 -> TrajType.SINGLEVALUE
            - lie on a Cartesian grid -> TrajType.ONGRID
            - none of the above -> TrajType.NOTONGRID


        Parameters
        ----------
            tolerance:
                absolute tolerance in checking if points are on integer grid positions

        Returns
        -------
            ((types along kz,ky,kx),(types along k2,k1,k0))

        # TODO: consider non-integer positions that are on a grid, e.g. (0.5, 1, 1.5, ....)
        """

        # Matrix describing trajectory-type [(kz, ky, kx), (k2, k1, k0)]
        # Start with everything not on a grid (arbitrary k-space locations).
        # We use the value of the enum-type to make it easier to do array operations.
        traj_type_matrix = torch.zeros(3, 3, dtype=torch.int)
        for ind, ks in enumerate((self.kz, self.ky, self.kx)):
            values_on_grid = not ks.is_floating_point() or torch.all(ks.frac() <= tolerance)
            for dim in (-3, -2, -1):
                if ks.shape[dim] == 1:
                    traj_type_matrix[ind, dim] |= TrajType.SINGLEVALUE.value | TrajType.ONGRID.value
                if values_on_grid:
                    traj_type_matrix[ind, dim] |= TrajType.ONGRID.value

        # kz should only have flags that are enabled in all columns
        # k2 only flags enabled in all rows, etc
        type_zyx = [TrajType(i.item()) for i in np.bitwise_and.reduce(traj_type_matrix, axis=1)]
        type_210 = [TrajType(i.item()) for i in np.bitwise_and.reduce(traj_type_matrix, axis=0)]

        # make mypy recognize return  will always have len=3
        return (type_zyx[0], type_zyx[1], type_zyx[2]), (type_210[0], type_210[1], type_210[2])

    @property
    def type_along_kzyx(self) -> tuple[TrajType, TrajType, TrajType]:
        """Type of trajectory along kz-ky-kx."""
        if self._expected_tensor_versions != (self.kz._version, self.ky._version, self.kx._version):
            raise NotImplementedError('The trajectory has been modified inplace. This is not supported')
        return self._type_along_kzyx

    @property
    def type_along_k210(self) -> tuple[TrajType, TrajType, TrajType]:
        """Type of trajectory along k2-k1-k0."""
        if self._expected_tensor_versions != (self.kz._version, self.ky._version, self.kx._version):
            raise NotImplementedError('The trajectory has been modified inplace. This is not supported')
        return self._type_along_k210

    def as_tensor(self, stack_dim=0) -> torch.Tensor:
        """Tensor representation of the trajectory.

        Parameters
        ----------
        stack_dim
            The dimension to stack the tensor along.
        """
        shape = self.broadcasted_shape
        return torch.stack([traj.expand(*shape) for traj in (self.kz, self.ky, self.kx)], dim=stack_dim)

    def __init__(
        self,
        kz: torch.Tensor,
        ky: torch.Tensor,
        kx: torch.Tensor,
        repeat_detection_tolerance: float | None = 1e-8,
        grid_detection_tolerance: float = 1e-3,
    ) -> None:
        """K-Space Trajectory dataclass.

        Reduces repeated dimensions to singletons if repeat_detection_tolerance
        is not set to None.

        Parameters
        ----------
        kz, ky, kx
            trajectory coordinates to set
        repeat_detection_tolerance
            Tolerance for repeat detection. Set to None to disable.
        grid_detection_tolerance
            tolerance to detect if trajectory points are on integer grid positions
        """
        if repeat_detection_tolerance is not None:
            kz, ky, kx = (remove_repeat(tensor, repeat_detection_tolerance) for tensor in (kz, ky, kx))

        # use of setattr due to frozen dataclass
        object.__setattr__(self, 'kz', kz)
        object.__setattr__(self, 'ky', ky)
        object.__setattr__(self, 'kx', kx)

        try:
            shape = self.broadcasted_shape
        except ValueError:
            raise ValueError('The k-space trajectory dimensions must be broadcastable.')
        if len(shape) != 4:
            raise ValueError('The k-space trajectory tensors should each have 4 dimensions.')

        type_along_kzyx, type_along_k210 = self._traj_types(grid_detection_tolerance)
        # use of setattr due to frozen dataclass
        object.__setattr__(self, '_expected_tensor_versions', (self.kz._version, self.ky._version, self.kx._version))
        object.__setattr__(self, '_type_along_kzyx', type_along_kzyx)
        object.__setattr__(self, '_type_along_k210', type_along_k210)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        stack_dim: int = 0,
        repeat_detection_tolerance: float | None = 1e-8,
        grid_detection_tolerance: float = 1e-3,
    ) -> KTrajectory:
        """Create a KTrajectory from a tensor representation of the trajectory.

        Reduces repeated dimensions to singletons if repeat_detection_tolerance
        is not set to None.


        Parameters
        ----------
        tensor
            The tensor representation of the trajectory.
            This should be a 5-dim tensor, with (kz,ky,kx) stacked in this order along stack_dim
        stack_dim
            The dimension in the tensor the directions have been stacked along.
        repeat_detection_tolerance
            detects if broadcasting can be used, i.e. if dimensions are repeated.
            Set to None to disable.
        grid_detection_tolerance
            tolerance to detect if trajectory points are on integer grid positions
        """

        kz, ky, kx = torch.unbind(tensor, dim=stack_dim)
        return cls(
            kz,
            ky,
            kx,
            repeat_detection_tolerance=repeat_detection_tolerance,
            grid_detection_tolerance=grid_detection_tolerance,
        )

    def to(self, *args, **kwargs) -> KTrajectory:
        """Perform dtype and/or device conversion of trajectory.

        A torch.dtype and torch.device are inferred from the arguments
        of self.to(*args, **kwargs). Please have a look at the
        documentation of torch.Tensor.to() for more details.
        """
        return KTrajectory(
            kz=self.kz.to(*args, **kwargs), ky=self.ky.to(*args, **kwargs), kx=self.kx.to(*args, **kwargs)
        )

    def cuda(
        self,
        device: torch.device | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> KTrajectory:
        """Create copy of trajectory in CUDA memory.

        Parameters
        ----------
        device
            The destination GPU device. Defaults to the current CUDA device.
        non_blocking
            If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        memory_format
            The desired memory format of returned Tensor.
        """
        return KTrajectory(
            kz=self.kz.cuda(
                device=device, non_blocking=non_blocking, memory_format=memory_format
            ),  # type: ignore [call-arg]
            ky=self.ky.cuda(
                device=device, non_blocking=non_blocking, memory_format=memory_format
            ),  # type: ignore [call-arg]
            kx=self.kx.cuda(
                device=device, non_blocking=non_blocking, memory_format=memory_format
            ),  # type: ignore [call-arg]
        )

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> KTrajectory:
        """Create copy of trajectory in CPU memory.

        Parameters
        ----------
        memory_format
            The desired memory format of returned Tensor.
        """
        return KTrajectory(
            kz=self.kz.cpu(memory_format=memory_format),  # type: ignore [call-arg]
            ky=self.ky.cpu(memory_format=memory_format),  # type: ignore [call-arg]
            kx=self.kx.cpu(memory_format=memory_format),  # type: ignore [call-arg]
        )
