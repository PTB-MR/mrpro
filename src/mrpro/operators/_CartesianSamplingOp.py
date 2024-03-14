"""Cartesian Sampling Operator."""

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

import torch
from einops import rearrange

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.data.enums import TrajType
from mrpro.operators import LinearOperator


class CartesianSamplingOp(LinearOperator):
    """Cartesian Sampling Operator.

    Puts the data on a Cartesian sampled grid based on the k-space
    trajectory.
    """

    def __init__(self, encoding_shape: SpatialDimension[int], traj: KTrajectory) -> None:
        """Initialize Sampling Operator class.

        Parameters
        ----------
        encoding_shape
            dimension of the encoded k-space
        traj
            the k-space trajectories where the frequencies are sampled
        """
        super().__init__()
        encoding_shape = SpatialDimension.from_xyz(encoding_shape)

        # Cache as these might take some time to compute
        traj_type_kzyx = traj.type_along_kzyx
        ktraj_tensor = traj.as_tensor()

        # If a dimension is irregular or singleton, we will not perform any reordering
        # in it, only dimensions on a cartesian grid will be reordered.
        if traj_type_kzyx[-1] == TrajType.ONGRID:  # kx
            kx_idx = ktraj_tensor[-1, ...].to(dtype=torch.int64) + encoding_shape.x // 2
        else:
            encoding_shape.x = ktraj_tensor.shape[-1]
            kx_idx = rearrange(torch.arange(ktraj_tensor.shape[-1]), 'kx->1 1 1 kx')

        if traj_type_kzyx[-2] == TrajType.ONGRID:  # ky
            ky_idx = ktraj_tensor[-2, ...].to(dtype=torch.int64) + encoding_shape.y // 2
        else:
            encoding_shape.y = ktraj_tensor.shape[-2]
            ky_idx = rearrange(torch.arange(ktraj_tensor.shape[-2]), 'ky->1 1 ky 1')

        if traj_type_kzyx[-3] == TrajType.ONGRID:  # kz
            kz_idx = ktraj_tensor[-3, ...].to(dtype=torch.int64) + encoding_shape.z // 2
        else:
            encoding_shape.z = ktraj_tensor.shape[-3]
            kz_idx = rearrange(torch.arange(ktraj_tensor.shape[-3]), 'kz->1 kz 1 1')

        # 1D indices into a flattened tensor.
        kidx = kz_idx * encoding_shape.y * encoding_shape.x + ky_idx * encoding_shape.x + kx_idx
        kidx = rearrange(kidx, '... kz ky kx -> ... 1 (kz ky kx)')
        self.register_buffer('_fft_idx', kidx)
        # we can skip the indexing if the data is already sorted
        self._needs_indexing = not torch.all(torch.diff(kidx) == 1)

        self._kshape = traj.broadcasted_shape
        self._encoding_shape = encoding_shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator which selects acquired k-space data from k-space.

        Parameters
        ----------
        x
            k-space with dimensions given by encoding_shape

        Returns
        -------
            k-space data in original (i.e. acquired) dimensions
        """
        if self._encoding_shape != SpatialDimension(*x.shape[-3:]):
            raise ValueError('k-space data shape mismatch')

        if not self._needs_indexing:
            return (x,)

        x_kflat = rearrange(x, '... coil k2_enc k1_enc k0_enc -> ... coil (k2_enc k1_enc k0_enc)')
        # take_along_dim does broadcast, so no need for extending here
        x_indexed = torch.take_along_dim(x_kflat, self._fft_idx, dim=-1)
        # reshape to (... other coil, k2, k1, k0)
        x_reshaped = x_indexed.reshape(x.shape[:-3] + self._kshape[-3:])

        return (x_reshaped,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator sorting data into the encoding_space matrix.

        Parameters
        ----------
        y
            k-space data in original (i.e. acquired) shape

        Returns
        -------
            k-space data sorted into encoding_space matrix
        """
        if self._kshape[-3:] != y.shape[-3:]:
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
            self._encoding_shape.z * self._encoding_shape.y * self._encoding_shape.x,
            dtype=y.dtype,
            device=y.device,
        ).scatter_(dim=-1, index=idx_expanded, src=y_kflat)

        # reshape to  ..., other, coil, k2_enc, k1_enc, k0_enc
        y_reshaped = y_scattered.reshape(
            *y.shape[:-3],
            self._encoding_shape.z,
            self._encoding_shape.y,
            self._encoding_shape.x,
        )

        return (y_reshaped,)
