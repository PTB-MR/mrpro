"""Cartesian Sampling Operators."""

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
from einops import repeat

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.data.enums import TrajType
from mrpro.operators import LinearOperator


class CartesianSamplingOp(LinearOperator):
    def __init__(
        self,
        encoding_shape: SpatialDimension[int],
        traj: KTrajectory,
    ) -> None:
        """Cartesian Sampling Operator class.

        Parameters
        ----------
        encoding_shape
            dimension of the encoded k-space
        traj
            the k-space trajectories where the frequencies are sampled
        """
        super().__init__()

        # Find dimensions of Cartesian sampling
        fft_dims = [dim for dim in (-3, -2, -1) if traj.traj_type_along_kzyx[dim] == TrajType.ONGRID]

        # Cartesian dimensions were found, create sorting index
        if len(fft_dims) > 0:
            ktraj_tensor = traj.as_tensor()
            if -1 in fft_dims:
                kx_idx = ktraj_tensor[-1, ...] + encoding_shape.x // 2
            else:
                encoding_shape.x = ktraj_tensor.shape[-1]
                kx_idx = torch.ones_like(ktraj_tensor[0, ...]) * rearrange(
                    torch.linspace(0, ktraj_tensor.shape[-1] - 1, ktraj_tensor.shape[-1]), 'kx->1 1 1 kx'
                )
            if -2 in fft_dims:
                ky_idx = ktraj_tensor[-2, ...] + encoding_shape.y // 2
            else:
                encoding_shape.y = ktraj_tensor.shape[-2]
                ky_idx = torch.ones_like(ktraj_tensor[0, ...]) * rearrange(
                    torch.linspace(0, ktraj_tensor.shape[-2] - 1, ktraj_tensor.shape[-2]), 'ky->1 1 ky 1'
                )
            if -3 in fft_dims:
                kz_idx = ktraj_tensor[-3, ...] + encoding_shape.z // 2
            else:
                encoding_shape.z = ktraj_tensor.shape[-3]
                kz_idx = torch.ones_like(ktraj_tensor[0, ...]) * rearrange(
                    torch.linspace(0, ktraj_tensor.shape[-3] - 1, ktraj_tensor.shape[-3]), 'kz->1 kz 1 1'
                )
            other_idx = torch.ones_like(ktraj_tensor[0, ...]) * rearrange(
                torch.linspace(0, ktraj_tensor.shape[1] - 1, ktraj_tensor.shape[1]), 'other->other 1 1 1'
            )
            kidx = (
                other_idx * encoding_shape.z * encoding_shape.y * encoding_shape.x
                + kz_idx * encoding_shape.y * encoding_shape.x
                + ky_idx * encoding_shape.x
                + kx_idx
            )
            kidx = repeat(
                kidx.to(dtype=torch.int64, device=traj.kx.device), 'other k2 k1 k0->other coil k2 k1 k0', coil=1
            )

            self._fft_idx = kidx
            self._fft_idx_full = torch.zeros(0)  # not None to satisfy mypy

        # Make sure sorting index is actually needed
        if torch.all(torch.diff(self._fft_idx.flatten()) == 1):
            self._fft_idx = torch.zeros(0)  # not None to satisfy mypy

        self._fft_dims = tuple(fft_dims)
        self._kshape = traj.broadcasted_shape
        self._encoding_shape = encoding_shape

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward operator which selects acquired k-space data from k-space.

        Parameters
        ----------
        y
            k-space with dimensions given by encoding_shape

        Returns
        -------
            k-space data in original (i.e. acquired) dimensions
        """
        if self._encoding_shape != SpatialDimension(*y.shape[-3:]):
            raise ValueError('k-space data shape missmatch')

        if len(self._fft_dims) > 0 and len(self._fft_idx) > 0:
            if len(self._fft_idx_full) == 0 or self._fft_idx_full.shape[1] != y.shape[1]:
                coil_idx = torch.ones(
                    *(y.shape[:2] + self._fft_idx.shape[2:]), dtype=torch.int64, device=y.device
                ) * rearrange(
                    torch.linspace(0, y.shape[1] - 1, y.shape[1], dtype=torch.int64, device=y.device),
                    'coils->1 coils 1 1 1',
                )
                other_idx = torch.ones_like(self._fft_idx, dtype=torch.int64, device=y.device) * rearrange(
                    torch.linspace(0, y.shape[0] - 1, y.shape[0], dtype=torch.int64, device=y.device),
                    'other->other 1 1 1 1',
                )
                self._fft_idx_full = (
                    self._fft_idx
                    + other_idx * y.shape[1]
                    + coil_idx * self._encoding_shape.z * self._encoding_shape.y * self._encoding_shape.x
                )

            return torch.take(y, self._fft_idx_full)
        else:
            return y

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint operator sorting data into the encoding_space matrix.

        Parameters
        ----------
        y
            k-space data in original (i.e. acquired shape)

        Returns
        -------
            k-space data sorted into encoding_space matrix
        """

        if self._kshape[1:] != y.shape[-3:]:
            raise ValueError('k-space data shape missmatch')

        if len(self._fft_dims) > 0 and len(self._fft_idx) > 0:
            if len(self._fft_idx_full) == 0 or self._fft_idx_full.shape[1] != y.shape[1]:
                coil_idx = torch.ones(
                    *(y.shape[:2] + self._fft_idx.shape[2:]), dtype=torch.int64, device=y.device
                ) * rearrange(
                    torch.linspace(0, y.shape[1] - 1, y.shape[1], dtype=torch.int64, device=y.device),
                    'coils->1 coils 1 1 1',
                )
                other_idx = torch.ones_like(self._fft_idx, dtype=torch.int64, device=y.device) * rearrange(
                    torch.linspace(0, y.shape[0] - 1, y.shape[0], dtype=torch.int64, device=y.device),
                    'other->other 1 1 1 1',
                )
                self._fft_idx_full = (
                    self._fft_idx
                    + other_idx * y.shape[1]
                    + coil_idx * self._encoding_shape.z * self._encoding_shape.y * self._encoding_shape.x
                )

            ysort = torch.zeros(
                *(y.shape[:2] + (self._encoding_shape.z, self._encoding_shape.y, self._encoding_shape.x)),
                dtype=y.dtype,
                device=y.device,
            )

            return ysort.put_(self._fft_idx_full, y, accumulate=True)
        else:
            return y
