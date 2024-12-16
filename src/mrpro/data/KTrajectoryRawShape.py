"""KTrajectoryRawShape dataclass."""

from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from typing_extensions import Literal, Self


@dataclass(slots=True, frozen=True)
class KTrajectoryRawShape(MoveDataMixin):
    """K-space trajectory shaped ((other*k2*k1),k0).

    Order of directions is always kz, ky, kx
    Shape of each of kx,ky,kz is ((other,k2,k1),k0) this means that e.g. slices, averages... have not yet been
    separated from the phase and slice encoding dimensions. The trajectory is in the same shape as the raw data in the
    raw data file.
    """

    kz: torch.Tensor
    """(other,k2,k1,k0), phase encoding direction k2 if Cartesian."""

    ky: torch.Tensor
    """(other,k2,k1,k0), phase encoding direction k1 if Cartesian."""

    kx: torch.Tensor
    """(other,k2,k1,k0), frequency encoding direction k0 if Cartesian."""

    repeat_detection_tolerance: None | float = 1e-3
    """tolerance for repeat detection. Set to None to disable."""

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        stack_dim: int = 0,
        axes_order: Literal['zxy', 'zyx', 'yxz', 'yzx', 'xyz', 'xzy'] = 'zyx',
        repeat_detection_tolerance: float | None = 1e-6,
        grid_detection_tolerance: float = 1e-3,
        encoding_matrix: SpatialDimension | None = None,
    ) -> Self:
        """Create a KTrajectoryRawShape from a tensor representation of the trajectory.

        Parameters
        ----------
        tensor
            The tensor representation of the trajectory.
            This should be a 5-dim tensor, with (kz,ky,kx) stacked in this order along stack_dim
        stack_dim
            The dimension in the tensor the directions have been stacked along.
        axes_order
            Order of the axes in the tensor. Our convention usually is zyx order.
        repeat_detection_tolerance
            detects if broadcasting can be used, i.e. if dimensions are repeated.
            Set to None to disable.
        encoding_matrix
            if an encoding matrix is supplied, the trajectory is rescaled to fit
            within the matrix. Otherwise, it is left as-is.
        """
        kz, ky, kx = (tensor.narrow(stack_dim, start=axes_order.index(axis), length=1) for axis in 'zyx')

        def normalize(k, encoding_size):
            max_abs_range = 2 * k.max().abs()
            if encoding_size == 1 or max_abs_range < 1e-6:
                # a single encoding point should be at zero
                # avoid division by zero
                return k.new_zeros()
            return k * (encoding_size / max_abs_range)

        if encoding_matrix is not None:
            kz = normalize(kz, encoding_matrix.z)
            ky = normalize(ky, encoding_matrix.y)
            kx = normalize(kx, encoding_matrix.x)

        return cls(
            kz,
            ky,
            kx,
            repeat_detection_tolerance=repeat_detection_tolerance,
        )

    def sort_and_reshape(
        self,
        sort_idx: np.ndarray,
        n_k2: int,
        n_k1: int,
    ) -> KTrajectory:
        """Resort and reshape the raw trajectory to KTrajectory.

        Parameters
        ----------
        sort_idx
            Index which defines how combined dimension (other k2 k1) needs to be sorted such that it can be separated
            into three separate dimensions using simple reshape operation. This information needs to be provided from
            kheader.acq_info.
        n_k2
            number of k2 points.
        n_k1
            number of k1 points.

        Returns
        -------
            KTrajectory with kx, ky and kz each in the shape (other k2 k1 k0).
        """
        # Resort and reshape
        kz = rearrange(self.kz[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)
        ky = rearrange(self.ky[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)
        kx = rearrange(self.kx[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)

        return KTrajectory(kz, ky, kx, repeat_detection_tolerance=self.repeat_detection_tolerance)
