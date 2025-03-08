"""KTrajectory dataclass."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import ismrmrd
import numpy as np
import torch
from typing_extensions import Self

from mrpro.data.enums import TrajType
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.reduce_repeat import reduce_repeat
from mrpro.utils.reshape import unsqueeze_at
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.typing import FileOrPath


@dataclass(slots=True, frozen=True)
class KTrajectory(MoveDataMixin):
    """K-space trajectory.

    Contains the trajectory in k-space along the three dimensions `kz`, `ky`, `kx`,
    i.e. describes where in k-space each data point was acquired.

    The shape of each of `kx`, `ky`, `kz` is `(*other, coils=1, k2, k1, k0)`,
    where `other` can span multiple dimensions.

    Example for 2D-Cartesian trajectories:

        - `kx` changes along `k0` and is frequency encoding,
        - `ky` changes along `k1` and is phase encoding
        - `kz` is zero with shape `(1, 1, 1, 1, 1)`
    """

    kz: torch.Tensor
    """Trajectory in z direction / phase encoding direction k2 if Cartesian. Shape `(*other, coils=1, k2, k1, k0)`"""

    ky: torch.Tensor
    """Trajectory in y direction / phase encoding direction k1 if Cartesian. Shape `(*other, coils=1, k2, k1, k0)`"""

    kx: torch.Tensor
    """Trajectory in x direction / phase encoding direction k0 if Cartesian. Shape `(*other, coils=1, k2, k1, k0)`"""

    grid_detection_tolerance: float = 1e-3
    """tolerance of how close trajectory positions have to be to integer grid points."""

    repeat_detection_tolerance: float | None = 1e-3
    """tolerance for repeat detection. Set to `None` to disable."""

    def __post_init__(self) -> None:
        """Reduce repeated dimensions to singletons."""

        def as_any_float(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.float() if not tensor.is_floating_point() else tensor

        if self.repeat_detection_tolerance is not None:
            kz, ky, kx = (
                as_any_float(reduce_repeat(tensor, self.repeat_detection_tolerance))
                for tensor in (self.kz, self.ky, self.kx)
            )
            # use of setattr due to frozen dataclass
            object.__setattr__(self, 'kz', kz)
            object.__setattr__(self, 'ky', ky)
            object.__setattr__(self, 'kx', kx)

        try:
            shape = self.broadcasted_shape
        except RuntimeError:
            raise ValueError('The k-space trajectory dimensions must be broadcastable.') from None

        if len(shape) < 5:
            raise ValueError('The k-space trajectory tensors should each have at least 5 dimensions.')

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        stack_dim: int = 0,
        axes_order: Literal['zxy', 'zyx', 'yxz', 'yzx', 'xyz', 'xzy'] = 'zyx',
        repeat_detection_tolerance: float | None = 1e-6,
        grid_detection_tolerance: float = 1e-3,
        scaling_matrix: SpatialDimension | None = None,
    ) -> Self:
        """Create a KTrajectory from a tensor representation of the trajectory.

        Reduces repeated dimensions to singletons if repeat_detection_tolerance is not set to `None`.

        Parameters
        ----------
        tensor
            The tensor representation of the trajectory.
            This should be a 5-dim tensor, with (`kz`, `ky`, `kx`) stacked in this order along `stack_dim`.
        stack_dim
            The dimension in the tensor along which the directions are stacked.
        axes_order
            The order of the axes in the tensor. The MRpro convention is 'zyx'.
        repeat_detection_tolerance
            Tolerance for detecting repeated dimensions (broadcasting).
            If trajectory points differ by less than this value, they are considered identical.
            Set to None to disable this feature.
        grid_detection_tolerance
            Tolerance for detecting whether trajectory points align with integer grid positions.
            This tolerance is applied after rescaling if `scaling_matrix` is provided.
        scaling_matrix
            If a scaling matrix is provided, the trajectory is rescaled to fit within
            the dimensions of the matrix. If not provided, the trajectory remains unchanged.

        """
        ks = tensor.unbind(dim=stack_dim)
        kz, ky, kx = (ks[axes_order.index(axis)] for axis in 'zyx')

        def rescale(k: torch.Tensor, size: float) -> torch.Tensor:
            max_abs_range = 2 * k.abs().max()
            if size < 2 or max_abs_range < 1e-6:
                # a single encoding point should be at zero
                # avoid division by zero
                return torch.zeros_like(k)
            return k * (size / max_abs_range)

        if scaling_matrix is not None:
            kz = rescale(kz, scaling_matrix.z)
            ky = rescale(ky, scaling_matrix.y)
            kx = rescale(kx, scaling_matrix.x)

        return cls(
            kz,
            ky,
            kx,
            repeat_detection_tolerance=repeat_detection_tolerance,
            grid_detection_tolerance=grid_detection_tolerance,
        )

    @classmethod
    def from_ismrmrd(
        cls,
        filename: FileOrPath,
        dataset_idx: int = -1,
        acquisition_filter_criterion: Callable[[ismrmrd.Acquisition], bool] | None = None,
        normalize: bool = False,
    ) -> Self:
        """Read a k-space trajectory from an ISMRMRD file.

        The trajectory is extracted from an ISMRMRD file. If the encoding matrix is set in the header,
        the trajectory is rescaled to fit within the dimensions of the matrix.

        Parameters
        ----------
        filename
            The path to the ISMRMRD file.
        dataset_idx
            The index of the dataset in the file.
        acquisition_filter_criterion
            A function that takes an ISMRMRD acquisition and returns a boolean.
            If provided, only acquisitions for which the function returns `True` are included in the trajectory.
            `None` includes all acquisitions.
        normalize
            Normalize the trajectory to the encoding matrix. Consider enabling for non-cartesian trajectories or
            non-normalized ISMRMRD trajectories.
        """
        with ismrmrd.File(filename, 'r') as file:
            datasets = list(file.keys())
            if not datasets:
                raise ValueError('No datasets found in the ISMRMRD file.')
            if not -len(datasets) <= dataset_idx < len(datasets):
                raise ValueError(f'Dataset index {dataset_idx} out of range, available datasets: {datasets}')
            dataset = file[datasets[dataset_idx]]

            acquisitions = [
                acq
                for acq in dataset.acquisitions
                if acquisition_filter_criterion is None or acquisition_filter_criterion(acq)
            ]
            if not acquisitions:
                raise ValueError('No matching acquisitions found in the ISMRMRD file.')
            traj = torch.stack([torch.as_tensor(acq.traj, dtype=torch.float32) for acq in acquisitions], dim=0)
            if not normalize:
                scaling_matrix = None
            elif (
                dataset.header.encoding
                and dataset.header.encoding[0].encodedSpace
                and dataset.header.encoding[0].encodedSpace.matrixSize
            ):
                scaling_matrix = SpatialDimension.from_xyz(dataset.header.encoding[0].encodedSpace.matrixSize)
            else:
                raise ValueError(
                    'Requested normalization, but the ISMRMRD file does not contain an encoding matrix size. '
                    'Consider adding it to the header.'
                )

        if traj.shape[-1] != 3:  # enforce 3D trajectory
            zero = torch.zeros_like(traj[..., :1])
            traj = torch.cat([traj, *([zero] * (3 - traj.shape[-1]))], dim=-1)
        traj = unsqueeze_at(traj, dim=-3, n=5 - traj.ndim + 1)  # +1 due to stack_dim

        return cls.from_tensor(traj, stack_dim=-1, axes_order='xyz', scaling_matrix=scaling_matrix)

    @property
    def broadcasted_shape(self) -> torch.Size:
        """The broadcasted shape of the trajectory.

        Returns
        -------
            broadcasted shape of trajectory
        """
        shape = torch.broadcast_shapes(self.kx.shape, self.ky.shape, self.kz.shape)
        return shape

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
            - are of shape 1 -> `TrajType.SINGLEVALUE`
            - lie on a Cartesian grid -> `TrajType.ONGRID`

        Parameters
        ----------
        tolerance:
            absolute tolerance in checking if points are on integer grid positions

        Returns
        -------
            (`(types along kz,ky,kx)`,`(types along k2,k1,k0)`)

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
        type_zyx = [TrajType(int(i)) for i in np.bitwise_and.reduce(traj_type_matrix.numpy(), axis=1)]
        type_210 = [TrajType(int(i)) for i in np.bitwise_and.reduce(traj_type_matrix.numpy(), axis=0)]

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
