"""Cartesian Sampling Operator."""

import torch
from einops import rearrange, repeat

from mrpro.data.enums import TrajType
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.LinearOperator import LinearOperator


class CartesianSamplingOp(LinearOperator):
    """Cartesian Sampling Operator.

    Selects points on a Cartesian grid based on the k-space trajectory.
    Thus, the adjoint sorts the data into regular Cartesian sampled grid based on the k-space
    trajectory. Non-acquired points are zero-filled.
    """

    def __init__(self, encoding_matrix: SpatialDimension[int], traj: KTrajectory) -> None:
        """Initialize Sampling Operator class.

        Parameters
        ----------
        encoding_matrix
            shape of the encoded k-space.
            Only values for directions in which the trajectory is Cartesian will be used
            in the adjoint to determine the shape after reordering,
            i.e., the operator's domain.
        traj
            the k-space trajectory describing at which frequencies data is sampled.
            Its broadcasted shape will be used to determine the shape after sampling,
            i.e., the operator's range
        """
        super().__init__()
        # the shape of the k data,
        sorted_grid_shape = SpatialDimension.from_xyz(encoding_matrix)

        # Cache as these might take some time to compute
        traj_type_kzyx = traj.type_along_kzyx
        ktraj_tensor = traj.as_tensor()

        # If a dimension is irregular or singleton, we will not perform any reordering
        # in it and the shape of data will remain.
        # only dimensions on a cartesian grid will be reordered.
        if traj_type_kzyx[-1] == TrajType.ONGRID:  # kx
            kx_idx = ktraj_tensor[-1, ...].round().to(dtype=torch.int64) + sorted_grid_shape.x // 2
        else:
            sorted_grid_shape.x = ktraj_tensor.shape[-1]
            kx_idx = repeat(torch.arange(ktraj_tensor.shape[-1]), 'k0->other k1 k2 k0', other=1, k2=1, k1=1)

        if traj_type_kzyx[-2] == TrajType.ONGRID:  # ky
            ky_idx = ktraj_tensor[-2, ...].round().to(dtype=torch.int64) + sorted_grid_shape.y // 2
        else:
            sorted_grid_shape.y = ktraj_tensor.shape[-2]
            ky_idx = repeat(torch.arange(ktraj_tensor.shape[-2]), 'k1->other k1 k2 k0', other=1, k2=1, k0=1)

        if traj_type_kzyx[-3] == TrajType.ONGRID:  # kz
            kz_idx = ktraj_tensor[-3, ...].round().to(dtype=torch.int64) + sorted_grid_shape.z // 2
        else:
            sorted_grid_shape.z = ktraj_tensor.shape[-3]
            kz_idx = repeat(torch.arange(ktraj_tensor.shape[-3]), 'k2->other k1 k2 k0', other=1, k1=1, k0=1)

        # 1D indices into a flattened tensor.
        kidx = kz_idx * sorted_grid_shape.y * sorted_grid_shape.x + ky_idx * sorted_grid_shape.x + kx_idx
        kidx = rearrange(kidx, '... kz ky kx -> ... 1 (kz ky kx)')
        self.register_buffer('_fft_idx', kidx)
        # we can skip the indexing if the data is already sorted
        self._needs_indexing = not torch.all(torch.diff(kidx) == 1)

        self._trajectory_shape = traj.broadcasted_shape
        self._sorted_grid_shape = sorted_grid_shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator which selects acquired k-space data from k-space.

        Parameters
        ----------
        x
            k-space, fully sampled (or zerofilled) and sorted in Cartesian dimensions
            with shape given by encoding_matrix

        Returns
        -------
            selected k-space data in acquired shape (as described by the trajectory)
        """
        if self._sorted_grid_shape != SpatialDimension(*x.shape[-3:]):
            raise ValueError('k-space data shape mismatch')

        if not self._needs_indexing:
            return (x,)

        x_kflat = rearrange(x, '... coil k2_enc k1_enc k0_enc -> ... coil (k2_enc k1_enc k0_enc)')
        # take_along_dim does broadcast, so no need for extending here
        x_indexed = torch.take_along_dim(x_kflat, self._fft_idx, dim=-1)
        # reshape to (... other coil, k2, k1, k0)
        x_reshaped = x_indexed.reshape(x.shape[:-3] + self._trajectory_shape[-3:])

        return (x_reshaped,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator sorting data into the encoding_space matrix.

        Parameters
        ----------
        y
            k-space data in acquired shape

        Returns
        -------
            k-space data sorted into encoding_space matrix
        """
        if self._trajectory_shape[-3:] != y.shape[-3:]:
            raise ValueError('k-space data shape mismatch')

        if not self._needs_indexing:
            return (y,)

        y_kflat = rearrange(y, '... coil k2 k1 k0 -> ... coil (k2 k1 k0)')

        # scatter does not broadcast, so we need to manually broadcast the indices
        broadcast_shape = torch.broadcast_shapes(self._fft_idx.shape[:-1], y_kflat.shape[:-1])
        idx_expanded = torch.broadcast_to(self._fft_idx, (*broadcast_shape, self._fft_idx.shape[-1]))

        # although scatter_ is inplace, this will not cause issues with autograd, as self
        # is always constant zero and gradients w.r.t. src work as expected.
        y_scattered = torch.zeros(
            *broadcast_shape,
            self._sorted_grid_shape.z * self._sorted_grid_shape.y * self._sorted_grid_shape.x,
            dtype=y.dtype,
            device=y.device,
        ).scatter_(dim=-1, index=idx_expanded, src=y_kflat)

        # reshape to  ..., other, coil, k2_enc, k1_enc, k0_enc
        y_reshaped = y_scattered.reshape(
            *y.shape[:-3],
            self._sorted_grid_shape.z,
            self._sorted_grid_shape.y,
            self._sorted_grid_shape.x,
        )

        return (y_reshaped,)
