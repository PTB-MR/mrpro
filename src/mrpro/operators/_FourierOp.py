"""Fourier Operator."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence

import numpy as np
import torch
from torchkbnufft import KbNufft
from torchkbnufft import KbNufftAdjoint

from mrpro.data._KTrajectory import KTrajectory
from mrpro.data._SpatialDimension import SpatialDimension
from mrpro.data.enums import TrajType
from mrpro.operators._FastFourierOp import FastFourierOp
from mrpro.operators._LinearOperator import LinearOperator


class FourierOp(LinearOperator):
    """Fourier Operator class."""

    def __init__(
        self,
        recon_matrix: SpatialDimension[int],
        encoding_matrix: SpatialDimension[int],
        traj: KTrajectory,
        nufft_oversampling: float = 2.0,
        nufft_numpoints: int = 6,
        nufft_kbwidth: float = 2.34,
    ) -> None:
        """Fourier Operator class.

        Parameters
        ----------
        recon_matrix
            dimension of the reconstructed image
        encoding_matrix
            dimension of the encoded k-space
        traj
            the k-space trajectories where the frequencies are sampled
        nufft_oversampling
            oversampling used for interpolation in non-uniform FFTs
        nufft_numpoints
            number of neighbors for interpolation in non-uniform FFTs
        nufft_kbwidth
            size of the Kaiser-Bessel kernel interpolation in non-uniform FFTs
        """
        super().__init__()

        def get_spatial_dims(spatial_dims: SpatialDimension, dims: Sequence[int]):
            return [
                s
                for s, i in zip((spatial_dims.z, spatial_dims.y, spatial_dims.x), (-3, -2, -1), strict=True)
                if i in dims
            ]

        def get_traj(traj: KTrajectory, dims: Sequence[int]):
            return [k for k, i in zip((traj.kz, traj.ky, traj.kx), (-3, -2, -1), strict=True) if i in dims]

        self._ignore_dims, self._fft_dims, self._nufft_dims = [], [], []
        for dim, type_ in zip((-3, -2, -1), traj.type_along_kzyx, strict=True):
            if type_ & TrajType.SINGLEVALUE:
                # dimension which do not require any transform
                self._ignore_dims.append(dim)
            elif type_ & TrajType.ONGRID:
                self._fft_dims.append(dim)
            else:
                self._nufft_dims.append(dim)

        if self._fft_dims:
            self._fast_fourier_op = FastFourierOp(
                dim=tuple(self._fft_dims),
                recon_matrix=get_spatial_dims(recon_matrix, self._fft_dims),
                encoding_matrix=get_spatial_dims(encoding_matrix, self._fft_dims),
            )

        # Find dimensions which require NUFFT
        if self._nufft_dims:
            fft_dims_k210 = [
                dim
                for dim in (-3, -2, -1)
                if (traj.type_along_k210[dim] & TrajType.ONGRID)
                and not (traj.type_along_k210[dim] & TrajType.SINGLEVALUE)
            ]
            if self._fft_dims != fft_dims_k210:
                raise NotImplementedError(
                    'If both FFT and NUFFT dims are present, Cartesian FFT dims need to be aligned with the '
                    'k-space dimension, i.e. kx along k0, ky along k1 and kz along k2',
                )

            self._nufft_im_size = get_spatial_dims(recon_matrix, self._nufft_dims)
            grid_size = [int(size * nufft_oversampling) for size in self._nufft_im_size]
            omega = [
                k * 2 * torch.pi / ks
                for k, ks in zip(
                    get_traj(traj, self._nufft_dims),
                    get_spatial_dims(encoding_matrix, self._nufft_dims),
                    strict=True,
                )
            ]

            # Broadcast shapes not always needed but also does not hurt
            omega = [k.expand(*np.broadcast_shapes(*[k.shape for k in omega])) for k in omega]
            self.register_buffer('_omega', torch.stack(omega, dim=-4))  # use the 'coil' dim for the direction

            self._fwd_nufft_op = KbNufft(
                im_size=self._nufft_im_size,
                grid_size=grid_size,
                numpoints=nufft_numpoints,
                kbwidth=nufft_kbwidth,
            )
            self._adj_nufft_op = KbNufftAdjoint(
                im_size=self._nufft_im_size,
                grid_size=grid_size,
                numpoints=nufft_numpoints,
                kbwidth=nufft_kbwidth,
            )

            self._kshape = traj.broadcasted_shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator mapping the coil-images to the coil k-space data.

        Parameters
        ----------
        x
            coil image data with shape: (... coils z y x)

        Returns
        -------
            coil k-space data with shape: (... coils k2 k1 k0)
        """
        if len(self._fft_dims):
            # FFT
            (x,) = self._fast_fourier_op.forward(x)

        if self._nufft_dims:
            # we need to move the nufft-dimensions to the end and flatten all other dimensions
            # so the new shape will be (... non_nufft_dims) coils nufft_dims
            # we could move the permute to __init__ but then we still would need to prepend if len(other)>1
            keep_dims = [-4, *self._nufft_dims]  # -4 is always coil
            permute = [i for i in range(-x.ndim, 0) if i not in keep_dims] + keep_dims
            unpermute = np.argsort(permute)

            x = x.permute(*permute)
            permuted_x_shape = x.shape
            x = x.flatten(end_dim=-len(keep_dims) - 1)

            # omega should be (... non_nufft_dims) n_nufft_dims (nufft_dims)
            # TODO: consider moving the broadcast along fft dimensions to __init__ (independent of x shape).
            omega = self._omega.permute(*permute)
            omega = omega.broadcast_to(*permuted_x_shape[: -len(keep_dims)], *omega.shape[-len(keep_dims) :])
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            x = self._fwd_nufft_op(x, omega, norm='ortho')

            shape_nufft_dims = [self._kshape[i] for i in self._nufft_dims]
            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], -1, *shape_nufft_dims)  # -1 is coils
            x = x.permute(*unpermute)
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator mapping the coil k-space data to the coil images.

        Parameters
        ----------
        x
            coil k-space data with shape: (... coils k2 k1 k0)

        Returns
        -------
            coil image data with shape: (... coils z y x)
        """
        if self._fft_dims:
            # IFFT
            (x,) = self._fast_fourier_op.adjoint(x)

        if self._nufft_dims:
            # we need to move the nufft-dimensions to the end, flatten them and flatten all other dimensions
            # so the new shape will be (... non_nufft_dims) coils (nufft_dims)
            keep_dims = [-4, *self._nufft_dims]  # -4 is coil
            permute = [i for i in range(-x.ndim, 0) if i not in keep_dims] + keep_dims
            unpermute = np.argsort(permute)

            x = x.permute(*permute)
            permuted_x_shape = x.shape
            x = x.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            omega = self._omega.permute(*permute)
            omega = omega.broadcast_to(*permuted_x_shape[: -len(keep_dims)], *omega.shape[-len(keep_dims) :])
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            x = self._adj_nufft_op(x, omega, norm='ortho')

            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], *x.shape[-len(keep_dims) :])
            x = x.permute(*unpermute)

        return (x,)
