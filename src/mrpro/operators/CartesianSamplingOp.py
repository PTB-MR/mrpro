"""Cartesian Sampling Operator."""

import warnings

import torch
from einops import rearrange, repeat

from mrpro.data.enums import TrajType
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils.reshape import unsqueeze_left


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
            kx_idx = repeat(torch.arange(ktraj_tensor.shape[-1]), 'k0->other k2 k1 k0', other=1, k2=1, k1=1)

        if traj_type_kzyx[-2] == TrajType.ONGRID:  # ky
            ky_idx = ktraj_tensor[-2, ...].round().to(dtype=torch.int64) + sorted_grid_shape.y // 2
        else:
            sorted_grid_shape.y = ktraj_tensor.shape[-2]
            ky_idx = repeat(torch.arange(ktraj_tensor.shape[-2]), 'k1->other k2 k1 k0', other=1, k2=1, k0=1)

        if traj_type_kzyx[-3] == TrajType.ONGRID:  # kz
            kz_idx = ktraj_tensor[-3, ...].round().to(dtype=torch.int64) + sorted_grid_shape.z // 2
        else:
            sorted_grid_shape.z = ktraj_tensor.shape[-3]
            kz_idx = repeat(torch.arange(ktraj_tensor.shape[-3]), 'k2->other k2 k1 k0', other=1, k1=1, k0=1)

        # 1D indices into a flattened tensor.
        kidx = kz_idx * sorted_grid_shape.y * sorted_grid_shape.x + ky_idx * sorted_grid_shape.x + kx_idx
        kidx = rearrange(kidx, '... kz ky kx -> ... 1 (kz ky kx)')

        # check that all points are inside the encoding matrix
        inside_encoding_matrix = (
            ((kx_idx >= 0) & (kx_idx < sorted_grid_shape.x))
            & ((ky_idx >= 0) & (ky_idx < sorted_grid_shape.y))
            & ((kz_idx >= 0) & (kz_idx < sorted_grid_shape.z))
        )
        if not torch.all(inside_encoding_matrix):
            warnings.warn(
                'K-space points lie outside of the encoding_matrix and will be ignored.'
                ' Increase the encoding_matrix to include these points.',
                stacklevel=2,
            )

            inside_encoding_matrix = rearrange(inside_encoding_matrix, '... kz ky kx -> ... 1 (kz ky kx)')
            inside_encoding_matrix_idx = inside_encoding_matrix.nonzero(as_tuple=True)[-1]
            inside_encoding_matrix_idx = torch.reshape(inside_encoding_matrix_idx, (*kidx.shape[:-1], -1))
            self.register_buffer('_inside_encoding_matrix_idx', inside_encoding_matrix_idx)
            kidx = torch.take_along_dim(kidx, inside_encoding_matrix_idx, dim=-1)
        else:
            self._inside_encoding_matrix_idx: torch.Tensor | None = None

        self.register_buffer('_fft_idx', kidx)

        # we can skip the indexing if the data is already sorted
        self._needs_indexing = (
            not torch.all(torch.diff(kidx) == 1)
            or traj.broadcasted_shape[-3:] != sorted_grid_shape.zyx
            or self._inside_encoding_matrix_idx is not None
        )

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
        # take_along_dim broadcasts, but needs the same number of dimensions
        idx = unsqueeze_left(self._fft_idx, x_kflat.ndim - self._fft_idx.ndim)
        x_inside_encoding_matrix = torch.take_along_dim(x_kflat, idx, dim=-1)

        if self._inside_encoding_matrix_idx is None:
            # all trajectory points are inside the encoding matrix
            x_indexed = x_inside_encoding_matrix
        else:
            # we need to add zeros
            x_indexed = self._broadcast_and_scatter_along_last_dim(
                x_inside_encoding_matrix,
                self._trajectory_shape[-1] * self._trajectory_shape[-2] * self._trajectory_shape[-3],
                self._inside_encoding_matrix_idx,
            )

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

        if self._inside_encoding_matrix_idx is not None:
            idx = unsqueeze_left(self._inside_encoding_matrix_idx, y_kflat.ndim - self._inside_encoding_matrix_idx.ndim)
            y_kflat = torch.take_along_dim(y_kflat, idx, dim=-1)

        y_scattered = self._broadcast_and_scatter_along_last_dim(
            y_kflat, self._sorted_grid_shape.z * self._sorted_grid_shape.y * self._sorted_grid_shape.x, self._fft_idx
        )

        # reshape to  ..., other, coil, k2_enc, k1_enc, k0_enc
        y_reshaped = y_scattered.reshape(
            *y.shape[:-3],
            self._sorted_grid_shape.z,
            self._sorted_grid_shape.y,
            self._sorted_grid_shape.x,
        )

        return (y_reshaped,)

    @staticmethod
    def _broadcast_and_scatter_along_last_dim(
        data_to_scatter: torch.Tensor, n_last_dim: int, scatter_index: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast scatter index and scatter into zero tensor.

        Parameters
        ----------
        data_to_scatter
            Data to be scattered at indices scatter_index
        n_last_dim
            Number of data points in last dimension
        scatter_index
            Indices describing where to scatter data

        Returns
        -------
            Data scattered into tensor along scatter_index
        """
        # scatter does not broadcast, so we need to manually broadcast the indices
        broadcast_shape = torch.broadcast_shapes(scatter_index.shape[:-1], data_to_scatter.shape[:-1])
        idx_expanded = torch.broadcast_to(scatter_index, (*broadcast_shape, scatter_index.shape[-1]))

        # although scatter_ is inplace, this will not cause issues with autograd, as self
        # is always constant zero and gradients w.r.t. src work as expected.
        data_scattered = torch.zeros(
            *broadcast_shape,
            n_last_dim,
            dtype=data_to_scatter.dtype,
            device=data_to_scatter.device,
        ).scatter_(dim=-1, index=idx_expanded, src=data_to_scatter)

        return data_scattered

    @property
    def gram(self) -> 'CartesianSamplingGramOp':
        """Return the Gram operator for this Cartesian Sampling Operator.

        Returns
        -------
            Gram operator for this Cartesian Sampling Operator
        """
        return CartesianSamplingGramOp(self)


class CartesianSamplingGramOp(LinearOperator):
    """Gram operator for Cartesian Sampling Operator.

    The Gram operator is the composition CartesianSamplingOp.H @ CartesianSamplingOp.
    """

    def __init__(self, sampling_op: CartesianSamplingOp):
        """Initialize Cartesian Sampling Gram Operator class.

        This should not be used directly, but rather through the `gram` method of a
        :class:`mrpro.operator.CartesianSamplingOp` object.

        Parameters
        ----------
        sampling_op
            The Cartesian Sampling Operator for which to create the Gram operator.
        """
        super().__init__()
        if sampling_op._needs_indexing:
            ones = torch.ones(*sampling_op._trajectory_shape[:-3], 1, *sampling_op._sorted_grid_shape.zyx)
            (mask,) = sampling_op.adjoint(*sampling_op.forward(ones))
            self.register_buffer('_mask', mask)
        else:
            self._mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the Gram operator.

        Parameters
        ----------
        x
            Input data

        Returns
        -------
            Output data
        """
        if self._mask is None:
            return (x,)
        return (x * self._mask,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the Gram operator.

        Parameters
        ----------
        y
            Input data

        Returns
        -------
            Output data
        """
        return self.forward(y)
