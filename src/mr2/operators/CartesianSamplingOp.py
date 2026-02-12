"""Cartesian Sampling Operator."""

import warnings

import torch
from einops import rearrange, repeat
from typing_extensions import Self

from mr2.data.enums import TrajType
from mr2.data.KTrajectory import KTrajectory
from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.LinearOperator import LinearOperator
from mr2.utils.reduce_repeat import reduce_repeat
from mr2.utils.reshape import unsqueeze_left


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
            i.e., the operator's range.
        """
        super().__init__()
        # the shape of the k data,
        sorted_grid_shape = SpatialDimension.from_xyz(encoding_matrix)

        # Cache as these might take some time to compute
        traj_type_kzyx = traj.type_along_kzyx
        traj_device = traj.device
        ktraj_tensor = traj.as_tensor()

        # If a dimension is irregular or singleton, we will not perform any reordering
        # in it and the shape of data will remain.
        # only dimensions on a cartesian grid will be reordered.
        # The device of the input trajectory is matched.
        if traj_type_kzyx[-1] == TrajType.ONGRID:  # kx
            kx_idx = ktraj_tensor[-1, ...].round().to(dtype=torch.int64) + sorted_grid_shape.x // 2
        else:
            sorted_grid_shape.x = ktraj_tensor.shape[-1]
            kx_idx = repeat(
                torch.arange(ktraj_tensor.shape[-1], device=traj_device),
                'k0->other coils k2 k1 k0',
                other=1,
                coils=1,
                k2=1,
                k1=1,
            )

        if traj_type_kzyx[-2] == TrajType.ONGRID:  # ky
            ky_idx = ktraj_tensor[-2, ...].round().to(dtype=torch.int64) + sorted_grid_shape.y // 2
        else:
            sorted_grid_shape.y = ktraj_tensor.shape[-2]
            ky_idx = repeat(
                torch.arange(ktraj_tensor.shape[-2], device=traj_device),
                'k1->other coils k2 k1 k0',
                other=1,
                coils=1,
                k2=1,
                k0=1,
            )

        if traj_type_kzyx[-3] == TrajType.ONGRID:  # kz
            kz_idx = ktraj_tensor[-3, ...].round().to(dtype=torch.int64) + sorted_grid_shape.z // 2
        else:
            sorted_grid_shape.z = ktraj_tensor.shape[-3]
            kz_idx = repeat(
                torch.arange(ktraj_tensor.shape[-3], device=traj_device),
                'k2->other coils k2 k1 k0',
                other=1,
                coils=1,
                k1=1,
                k0=1,
            )

        # 1D indices into a flattened tensor.
        kidx = kz_idx * sorted_grid_shape.y * sorted_grid_shape.x + ky_idx * sorted_grid_shape.x + kx_idx
        kidx = rearrange(kidx, '... kz ky kx -> ... (kz ky kx)')

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

            inside_encoding_matrix = rearrange(inside_encoding_matrix, '... kz ky kx -> ... (kz ky kx)')
            inside_encoding_matrix_idx = inside_encoding_matrix.nonzero(as_tuple=True)[-1]
            inside_encoding_matrix_idx = torch.reshape(inside_encoding_matrix_idx, (*kidx.shape[:-1], -1))
            self._inside_encoding_matrix_idx: torch.Tensor | None = inside_encoding_matrix_idx
            kidx = torch.take_along_dim(kidx, inside_encoding_matrix_idx, dim=-1)
        else:
            self._inside_encoding_matrix_idx = None

        self._fft_idx = kidx

        # we can skip the indexing if the data is already sorted
        self._needs_indexing = (
            not torch.all(torch.diff(kidx) == 1)
            or traj.shape[-3:] != sorted_grid_shape.zyx
            or self._inside_encoding_matrix_idx is not None
        )

        self._trajectory_shape = traj.shape
        self._sorted_grid_shape = sorted_grid_shape

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator which selects acquired k-space data from k-space.

        Parameters
        ----------
        x
            k-space, fully sampled (or zerofilled) and sorted in Cartesian dimensions
            with shape given by encoding_matrix.

        Returns
        -------
            Selected k-space data in acquired shape (as described by the trajectory).
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of CartesianSamplingOp.

        .. note::
            Prefer calling the instance of the CartesianSamplingOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if self._sorted_grid_shape != SpatialDimension(*x.shape[-3:]):
            raise ValueError('k-space data shape mismatch')

        if not self._needs_indexing:
            return (x,)

        x_kflat = rearrange(x, '... coils k2_enc k1_enc k0_enc -> ... coils (k2_enc k1_enc k0_enc)')
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

        # reshape to (... other coils, k2, k1, k0)
        x_reshaped = x_indexed.reshape(x.shape[:-3] + self._trajectory_shape[-3:])

        return (x_reshaped,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator sorting data into the encoding_space matrix.

        Parameters
        ----------
        y
            k-space data in acquired shape.

        Returns
        -------
            k-space data sorted into encoding_space matrix. Non-acquired points are zero-filled.
        """
        if self._trajectory_shape[-3:] != y.shape[-3:]:
            raise ValueError('k-space data shape mismatch')

        if not self._needs_indexing:
            return (y,)

        y_kflat = rearrange(y, '... coils k2 k1 k0 -> ... coils (k2 k1 k0)')

        if self._inside_encoding_matrix_idx is not None:
            idx = unsqueeze_left(self._inside_encoding_matrix_idx, y_kflat.ndim - self._inside_encoding_matrix_idx.ndim)
            y_kflat = torch.take_along_dim(y_kflat, idx, dim=-1)

        y_scattered = self._broadcast_and_scatter_along_last_dim(
            y_kflat, self._sorted_grid_shape.z * self._sorted_grid_shape.y * self._sorted_grid_shape.x, self._fft_idx
        )

        # reshape to  ..., other, coils, k2_enc, k1_enc, k0_enc
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
    def gram(self) -> 'CartesianMaskingOp':
        """Return the Gram operator for this Cartesian Sampling Operator.

        Returns
        -------
            Gram operator for this Cartesian Sampling Operator.
        """
        return CartesianMaskingOp.from_sampling_op(self)

    def __repr__(self) -> str:
        """Representation method for CartesianSamplingOperator."""
        device = self._fft_idx.device if self._fft_idx is not None else 'none'
        if self._inside_encoding_matrix_idx is None:
            enc_matrix_warning = ''
        else:
            enc_matrix_warning = (
                '\nk-space points lie outside of the encoding_matrix and will be ignored.'
                '\nIncrease the encoding_matrix to include these points.'
            )

        out = (
            f'{type(self).__name__} on device: {device}\n'
            f'Needs indexing: {self._needs_indexing}\n'
            f'Sorted grid shape: {self._sorted_grid_shape}'
            f'{enc_matrix_warning}'
        )
        return out


class CartesianMaskingOp(LinearOperator):
    """Cartesian Masking Operator.

    The Cartesian Masking Operator is the composition `CartesianSamplingOp.H @ CartesianSamplingOp`,
    which sets to zero all non sampled Cartesian k-space points.
    """

    def __init__(self, mask: torch.Tensor | None):
        """Initialize Cartesian Sampling Masking Operator from a mask.

        Parameters
        ----------
        mask
            The mask to use for the Cartesian Masking Operator.

        """
        super().__init__()
        self.mask = None if mask is None else reduce_repeat(mask.float())

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the Gram operator (Cartesian Masking).

        This operator applies a mask to the input tensor, effectively zeroing out
        points that were not sampled according to the Cartesian trajectory.
        It represents `CartesianSamplingOp.H @ CartesianSamplingOp`.

        Parameters
        ----------
        x
            Input k-space data, typically fully sampled or zero-filled.

        Returns
        -------
            Masked k-space data.
        """
        return super().__call__(x)

    @classmethod
    def from_trajectory(cls, traj: KTrajectory, encoding_matrix: SpatialDimension[int]) -> 'CartesianMaskingOp':
        """Initialize Cartesian Sampling Masking Operator from a trajectory.

        Parameters
        ----------
        traj
            The trajectory to use for the Cartesian Masking Operator.
        encoding_matrix
            The encoding matrix to use for the Cartesian Masking Operator.
        """
        return cls.from_sampling_op(CartesianSamplingOp(encoding_matrix, traj))

    @classmethod
    def from_sampling_op(cls, sampling_op: CartesianSamplingOp) -> Self:
        """Initialize Cartesian Sampling Masking Operator from a Cartesian Sampling Operator.

        Parameters
        ----------
        sampling_op
            The Cartesian Sampling Operator for which to create the Gram operator.
        """
        if sampling_op._needs_indexing:
            ones = torch.ones(
                *sampling_op._trajectory_shape[:-3],
                *sampling_op._sorted_grid_shape.zyx,
                device=sampling_op._fft_idx.device,
            )
            (mask,) = sampling_op.adjoint(*sampling_op.forward(ones))
        else:
            mask = None
        return cls(mask)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of CartesianMaskingOp.

        .. note::
            Prefer calling the instance of the CartesianMaskingOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if self.mask is None:
            return (x,)
        return (x * self.mask,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the Gram operator (Cartesian Masking).

        Since the Cartesian Masking operator is self-adjoint (it involves
        applying a mask, which is a real-valued multiplication), its adjoint
        is the same as its forward operation.

        Parameters
        ----------
        y
            Input k-space data.

        Returns
        -------
            Masked k-space data.
        """
        return self.forward(y)

    @property
    def H(self) -> Self:  # noqa: N802
        """Return the adjoint of the Cartesian Masking Operator.

        Returns
        -------
            the same operator, as the masking operator is self-adjoint.
        """
        return self
