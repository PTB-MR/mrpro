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

    def _type_of_traj(self) -> None:
        """Calculate type of trajectory.

        This function checks if the entries of the trajectory along certain dimensions
            - are of shape 1 -> TrajType.SINGLEVALUE
            - lie on a Cartesian grid -> TrajType.ONGRID
            - none of the above -> TrajType.NOTONGRID

        Returns
        -------
            type of trajectory along kz-ky-kx and k2-k1-k0
        """
        # Check if trajectory has changed and type information needs to be updated
        if self._traj_version != [self.kz._version, self.ky._version, self.kx._version]:
            # Matrix describing trajectory-type [(kz, ky, kx), (k2, k1, k0)]
            # Start with everything not on a grid (arbitrary k-space locations).
            # We use the value of the enum-type to make it easier to do array operations.
            traj_type_matrix = torch.tensor([[TrajType.NOTONGRID.value] * 3] * 3, dtype=torch.int8)

            for ind, ks in enumerate((self.kz, self.ky, self.kx)):
                are_values_on_grid = (ks - torch.round(ks)) <= self.type_tolerance

                # Only True if True for all entries
                are_values_on_grid = torch.all(are_values_on_grid)

                for dim in (-3, -2, -1):
                    # Check if it is a singleton dimension
                    if ks.shape[dim] == 1:
                        traj_type_matrix[ind, dim] = TrajType.SINGLEVALUE.value
                    else:
                        if are_values_on_grid:
                            traj_type_matrix[ind, dim] = TrajType.ONGRID.value

            # Return trajectory-type
            def get_traj_type(type_matrix):
                traj_type = [TrajType.ONGRID.value] * 3
                for dim in (-3, -2, -1):
                    if torch.all(type_matrix[dim, :] == TrajType.SINGLEVALUE.value):
                        traj_type[dim] = TrajType.SINGLEVALUE.value
                    elif torch.any(type_matrix[dim, :] == TrajType.NOTONGRID.value):
                        traj_type[dim] = TrajType.NOTONGRID.value
                return traj_type

            # Default is to sum along (k2, k1, k0) to return the trajectory type for kz, ky and kx
            self._type_kz_ky_kx = [TrajType(val) for val in get_traj_type(traj_type_matrix)]
            # If we want instead the values along k2, k1, and k0 then we have to transpose the matrix.
            self._type_k2_k1_k0 = [
                TrajType(val) for val in get_traj_type(torch.transpose(traj_type_matrix, dim0=0, dim1=1))
            ]

            # Update version to remember current trajectory
            self._traj_version = [self.kz._version, self.ky._version, self.kx._version]

    @property
    def type_kz(self):
        """Type of trajectory along kz."""
        self._type_of_traj()
        return self._type_kz_ky_kx[-3]

    @property
    def type_ky(self):
        """Type of trajectory along ky."""
        self._type_of_traj()
        return self._type_kz_ky_kx[-2]

    @property
    def type_kx(self):
        """Type of trajectory along kx."""
        self._type_of_traj()
        return self._type_kz_ky_kx[-1]

    @property
    def traj_type_along_kzyx(self, tolerance: float = 1e-2) -> list[TrajType]:
        """Type of trajectory along kz-ky-kx.

        Parameters
        ----------
        tolerance
            tolerance used to determine if trajectory points lie on a gird

        Returns
        -------
            type of trajectory along kz-ky-kx
        """
        return self._type_kz_ky_kx

    @property
    def traj_type_along_k210(self, tolerance: float = 1e-2) -> list[TrajType]:
        """Type of trajectory along k2-k1-k0.

        Parameters
        ----------
        tolerance
            tolerance used to determine if trajectory points lie on a gird

        Returns
        -------
            type of trajectory along k2-k1-k0
        """
        return self._type_k2_k1_k0

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
        type_tolerance: float = 1e-2,
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
        """
        if repeat_detection_tolerance is not None:
            kz, ky, kx = (remove_repeat(tensor, repeat_detection_tolerance) for tensor in (kz, ky, kx))

        self.kz = kz
        self.ky = ky
        self.kx = kx

        # Calculate what the type of the trajectory is along different directions
        self.type_tolerance = type_tolerance
        self._traj_version = [-1, -1, -1]
        self._type_of_traj()

        try:
            shape = self.broadcasted_shape
        except ValueError:
            raise ValueError('The k-space trajectory dimensions must be broadcastable.')
        if len(shape) < 4:
            raise ValueError('The k-space trajectory tensors should each have at least 4 dimensions.')

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, stack_dim: int = 0, repeat_detection_tolerance: float | None = 1e-8
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
        """

        kz, ky, kx = torch.unbind(tensor, dim=stack_dim)
        return cls(kz, ky, kx, repeat_detection_tolerance=repeat_detection_tolerance)

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
