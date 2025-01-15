"""KTrajectoryRawShape dataclass."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from typing_extensions import Self

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension


@dataclass(slots=True, frozen=True)
class KTrajectoryRawShape(MoveDataMixin):
    """K-space trajectory shaped `((other*k2*k1), k0)`.

    Contains the k-space trajectory, i.e. a description of where data point was acquired in k-space,
    in the raw shape as it is read from the data file, before any reshaping or sorting by indices is applied.
    The shape of each of `kx`, `ky`,` kz` is `((other*k2*k1), k0)`,
    this means that e.g. slices, averages... have not yet been separated from the phase and slice encoding dimensions.
    """

    kz: torch.Tensor
    """`(other*k2*k1,k0)`, phase encoding direction k2 if Cartesian."""

    ky: torch.Tensor
    """`(other*k2*k1,k0)`, phase encoding direction k1 if Cartesian."""

    kx: torch.Tensor
    """`(other*k2*k1,k0),` frequency encoding direction k0 if Cartesian."""

    repeat_detection_tolerance: None | float = 1e-3
    """tolerance for repeat detection. Set to `None` to disable."""

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        stack_dim: int = 0,
        axes_order: Literal['zxy', 'zyx', 'yxz', 'yzx', 'xyz', 'xzy'] = 'zyx',
        repeat_detection_tolerance: float | None = 1e-6,
        scaling_matrix: SpatialDimension | None = None,
    ) -> Self:
        """Create a KTrajectoryRawShape from a tensor representation of the trajectory.

        Parameters
        ----------
        tensor
            The tensor representation of the trajectory.
            This should be a 5-dim tensor, with (kz, ky, kx) stacked in this order along `stack_dim`.
        stack_dim
            The dimension in the tensor along which the directions are stacked.
        axes_order
            The order of the axes in the tensor. The MRpro convention is 'zyx'.
        repeat_detection_tolerance
            Tolerance for detecting repeated dimensions (broadcasting).
            If trajectory points differ by less than this value, they are considered identical.
            Set to `None` to disable this feature.
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

        return cls(kz, ky, kx, repeat_detection_tolerance=repeat_detection_tolerance)

    def sort_and_reshape(
        self,
        sort_idx: np.ndarray,
        n_k2: int,
        n_k1: int,
    ) -> KTrajectory:
        """Resort and reshape the raw trajectory to KTrajectory.

        This function is used to sort the raw trajectory and reshape it to an `mrpro.data.KTrajectory`
        by separating the combined dimension `(other k2 k1)` into three separate dimensions.

        Parameters
        ----------
        sort_idx
            Index which defines how combined dimension `(other k2 k1)` needs to be sorted such that it can be separated
            into three separate dimensions using a reshape operation.
        n_k2
            number of k2 points.
        n_k1
            number of k1 points.

        Returns
        -------
            KTrajectory with kx, ky and kz each in the shape `(other k2 k1 k0)`.
        """
        # Resort and reshape
        kz = rearrange(self.kz[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)
        ky = rearrange(self.ky[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)
        kx = rearrange(self.kx[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)

        return KTrajectory(kz, ky, kx, repeat_detection_tolerance=self.repeat_detection_tolerance)
