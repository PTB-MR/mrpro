"""KTrajectory dataclass."""

from dataclasses import dataclass

import numpy as np
import torch
from typing_extensions import Self

from mrpro.data.enums import TrajType
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.utils import remove_repeat
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues


@dataclass(slots=True, frozen=True)
class KTrajectory(MoveDataMixin):
    """K-space trajectory.

    Order of directions is always kz, ky, kx
    Shape of each of kx,ky,kz is (other,k2,k1,k0)

    Example for 2D-Cartesian Trajectories:
        kx changes along k0 and is Frequency Encoding
        ky changes along k2 and is Phase Encoding
        kz is zero(1,1,1,1)
    """

    kz: torch.Tensor
    """Trajectory in z direction / phase encoding direction k2 if Cartesian. Shape (other,k2,k1,k0)"""

    ky: torch.Tensor
    """Trajectory in y direction / phase encoding direction k1 if Cartesian. Shape (other,k2,k1,k0)"""

    kx: torch.Tensor
    """Trajectory in x direction / phase encoding direction k0 if Cartesian. Shape (other,k2,k1,k0)"""

    grid_detection_tolerance: float = 1e-3
    """tolerance of how close trajectory positions have to be to integer grid points."""

    repeat_detection_tolerance: float | None = 1e-3
    """tolerance for repeat detection. Set to None to disable."""

    def __post_init__(self) -> None:
        """Reduce repeated dimensions to singletons."""

        def as_any_float(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.float() if not tensor.is_floating_point() else tensor

        if self.repeat_detection_tolerance is not None:
            kz, ky, kx = (
                as_any_float(remove_repeat(tensor, self.repeat_detection_tolerance))
                for tensor in (self.kz, self.ky, self.kx)
            )
            # use of setattr due to frozen dataclass
            object.__setattr__(self, 'kz', kz)
            object.__setattr__(self, 'ky', ky)
            object.__setattr__(self, 'kx', kx)

        try:
            shape = self.broadcasted_shape
        except ValueError:
            raise ValueError('The k-space trajectory dimensions must be broadcastable.') from None

        if len(shape) < 4:
            raise ValueError('The k-space trajectory tensors should each have at least 4 dimensions.')

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        stack_dim: int = 0,
        repeat_detection_tolerance: float | None = 1e-8,
        grid_detection_tolerance: float = 1e-3,
    ) -> Self:
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

    @property
    def broadcasted_shape(self) -> tuple[int, ...]:
        """The broadcasted shape of the trajectory.

        Returns
        -------
            broadcasted shape of trajectory
        """
        shape = np.broadcast_shapes(self.kx.shape, self.ky.shape, self.kz.shape)
        return tuple(shape)

    @property
    def type_along_kzyx(self) -> tuple[TrajType, TrajType, TrajType]:
        """Type of trajectory along kz-ky-kx."""
        return self._traj_types(self.grid_detection_tolerance)[0]

    @property
    def type_along_k210(self) -> tuple[TrajType, TrajType, TrajType]:
        """Type of trajectory along k2-k1-k0."""
        return self._traj_types(self.grid_detection_tolerance)[1]

    def _traj_types(
        self,
        tolerance: float,
    ) -> tuple[tuple[TrajType, TrajType, TrajType], tuple[TrajType, TrajType, TrajType]]:
        """Calculate the trajectory type along kzkykx and k2k1k0.

        Checks if the entries of the trajectory along certain dimensions
            - are of shape 1 -> TrajType.SINGLEVALUE
            - lie on a Cartesian grid -> TrajType.ONGRID

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
            values_on_grid = not ks.is_floating_point() or torch.all((ks - ks.round()).abs() <= tolerance)
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

    def as_tensor(self, stack_dim: int = 0) -> torch.Tensor:
        """Tensor representation of the trajectory.

        Parameters
        ----------
        stack_dim
            The dimension to stack the tensor along.
        """
        shape = self.broadcasted_shape
        return torch.stack([traj.expand(*shape) for traj in (self.kz, self.ky, self.kx)], dim=stack_dim)

    def __repr__(self):
        """Representation method for KTrajectory class."""
        z = summarize_tensorvalues(torch.tensor(self.kz.shape))
        y = summarize_tensorvalues(torch.tensor(self.ky.shape))
        x = summarize_tensorvalues(torch.tensor(self.kx.shape))
        out = f'{type(self).__name__} with shape: kz={z}, ky={y}, kx={x}'
        return out
