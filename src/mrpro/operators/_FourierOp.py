"""Fourier Operator."""

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

import numpy as np
import torch
from einops import rearrange
from einops import repeat
from torchkbnufft import KbNufft
from torchkbnufft import KbNufftAdjoint

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.data.enums import TrajType
from mrpro.operators import FastFourierOp
from mrpro.operators import LinearOperator


class FourierOp(LinearOperator):
    def __init__(
        self,
        recon_shape: SpatialDimension[int],
        encoding_shape: SpatialDimension[int],
        traj: KTrajectory,
        oversampling: SpatialDimension[float] = SpatialDimension(
            2.0, 2.0, 2.0
        ),  # TODO: maybe not the nicest, since only used for nuFFT
        numpoints: int = 6,
        kbwidth: float = 2.34,
    ) -> None:
        """Fourier Operator class.

        Parameters
        ----------
        recon_shape
            dimension of the reconstructed image
        encoding_shape
            dimension of the encoded k-space
        traj
            the k-space trajectories where the frequencies are sampled
        oversampling
            oversampling for (potential) nuFFT directions
        numpoints
            number of neighbors for interpolation for nuFFTs
        kbwidth
            size of the Kaiser-Bessel kernel for the nuFFT
        """

        super().__init__()

        def get_dims_for_traj_type(traj_type):
            return [dim for dim in (-3, -2, -1) if traj.traj_type_along_kzyx[dim] == traj_type]

        def get_spatial_dims(spatial_dims, dims):
            return [s for s, i in zip((spatial_dims.z, spatial_dims.y, spatial_dims.x), (-3, -2, -1)) if i in dims]

        def get_traj(traj, dims):
            return [k for k, i in zip((traj.kz, traj.ky, traj.kx), (-3, -2, -1)) if i in dims]

        # Find dimensions which do not require any transform
        self._ignore_dims = get_dims_for_traj_type(TrajType.SINGLEVALUE)

        # Find dimensions which require FFT
        self._fft_dims = get_dims_for_traj_type(TrajType.ONGRID)
        if len(self._fft_dims) > 0:
            self._fast_fourier_op = FastFourierOp(
                dim=self._fft_dims,
                recon_shape=get_spatial_dims(recon_shape, self._fft_dims),
                encoding_shape=get_spatial_dims(encoding_shape, self._fft_dims),
            )

        # Find dimensions which require NUFFT
        self._nufft_dims = get_dims_for_traj_type(TrajType.NOTONGRID)
        if len(self._nufft_dims) > 0:
            # Special case when the fft dimension does not align with the corresponding k2, k1 or k0 dimension.
            # E.g. kx is of shape (1,1,30,1,1): kx sampling along k2.
            if self._fft_dims != [dim for dim in (-3, -2, -1) if traj.traj_type_along_k210[dim] == TrajType.ONGRID]:
                raise NotImplementedError(
                    'Cartesian FFT dims need to be aligned with the k-space dimension,'
                    'i.e. kx along k0, ky along k1 and kz along k2'
                )

            self._nufft_im_size = get_spatial_dims(recon_shape, self._nufft_dims)
            grid_size = [
                int(s * os) for s, os in zip(self._nufft_im_size, get_spatial_dims(oversampling, self._nufft_dims))
            ]
            omega = [
                k * 2 * torch.pi / ks
                for k, ks in zip(get_traj(traj, self._nufft_dims), get_spatial_dims(encoding_shape, self._nufft_dims))
            ]

            # Broadcast shapes (not always needed but also does not hurt)
            omega_shapes = tuple([tuple(k.shape) for k in omega])
            omega = [k.expand(*np.broadcast_shapes(*omega_shapes)) for k in omega]

            # Bring to correct shape
            self._omega = torch.stack([k.flatten(start_dim=-3) for k in omega], dim=-2)
            self._fwd_nufft_op = KbNufft(
                im_size=self._nufft_im_size, grid_size=grid_size, numpoints=numpoints, kbwidth=kbwidth
            )
            self._adj_nufft_op = KbNufftAdjoint(
                im_size=self._nufft_im_size, grid_size=grid_size, numpoints=numpoints, kbwidth=kbwidth
            )

            # Determine how data needs to be reshaped to ensure nufft can be applied to the last dimension
            def get_dim_str_for_traj_type(traj_types, type):
                dims = [
                    dim_str for dim, dim_str in zip((-3, -2, -1), ('dim2', 'dim1', 'dim0')) if traj_types[dim] == type
                ]
                return ' '.join(dims)

            nufft_dims_str = get_dim_str_for_traj_type(traj.traj_type_along_kzyx, TrajType.NOTONGRID)
            fft_dims_str = get_dim_str_for_traj_type(traj.traj_type_along_kzyx, TrajType.ONGRID)
            ignore_dims_str = get_dim_str_for_traj_type(traj.traj_type_along_kzyx, TrajType.SINGLEVALUE)
            self._target_pattern = f'(other {fft_dims_str} {ignore_dims_str}) coils {nufft_dims_str}'

        self._kshape = traj.broadcasted_shape
        self._recon_shape = recon_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward operator mapping the coil-images to the coil k-space data.

        Parameters
        ----------
        x
            coil image data with shape: (other coils z y x)

        Returns
        -------
            coil k-space data with shape: (other coils k2 k1 k0)
        """
        if self._recon_shape != SpatialDimension(*x.shape[-3:]):
            raise ValueError('image data shape missmatch')

        if len(self._fft_dims) != 0:
            x = self._fast_fourier_op.forward(x)

        if len(self._nufft_dims) != 0:
            init_pattern = 'other coils dim2 dim1 dim0'

            # get shape before applying nuFFT
            nb, nc, _, _, _ = x.shape
            x = rearrange(x, init_pattern + '->' + self._target_pattern)

            # apply nuFFT
            if nb > 1 and len(self._fft_dims) >= 1 and len(self._nufft_dims) >= 1:
                # if multiple batches, repeat k-space trajectories
                nb_ = int(x.shape[0] / nb)
                omega = repeat(
                    self._omega, 'other n_nufft_dims nk -> (other other_rep) n_nufft_dims nk', other=nb, other_rep=nb_
                )
            else:
                omega = self._omega
            x = self._fwd_nufft_op(x, omega, norm='ortho')

            # identify separate dimensionality of nuFFT-dimensions
            nufft_dim_size = tuple([nk for nk, dim in zip(self._kshape[1:], (-3, -2, -1)) if dim in self._nufft_dims])

            # unflatten the nuFFT-dimensions
            x = x.reshape(*x.shape[:2], *nufft_dim_size)

            # bring to shape defined by k-space trajectories
            nk2, nk1, nk0 = self._kshape[1:]
            x = rearrange(
                x, self._target_pattern + '->' + init_pattern, other=nb, coils=nc, dim1=nk1, dim2=nk2, dim0=nk0
            )

        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint operator mapping the coil k-space data to the coil images.

        Parameters
        ----------
        y
            coil k-space data with shape: (other coils k2 k1 k0)

        Returns
        -------
            coil image data with shape: (other coils z y x)
        """

        # apply IFFT
        if len(self._fft_dims) != 0:
            y = self._fast_fourier_op.adjoint(y)

        # move dim where FFT was already performed such nuFFT can be performed
        if len(self._nufft_dims) != 0:
            init_pattern = 'other coils dim2 dim1 dim0'

            # get shape before applying nuFFT
            nb, nc, _, _, _ = y.shape
            y = rearrange(y, init_pattern + '->' + self._target_pattern)

            # flatten the nuFFT-dimensions
            nufft_dim_size = tuple([nk for nk, dim in zip(self._kshape[1:], (-3, -2, -1)) if dim in self._nufft_dims])
            y_shape_tuple = tuple(y.shape)
            nk = int(torch.prod(torch.tensor(nufft_dim_size)))
            y = y.reshape(*y_shape_tuple[:2], nk)

            # apply adjoint nuFFT
            if nb > 1 and len(self._fft_dims) >= 1 and len(self._nufft_dims) >= 1:
                # if multiple batches, repeat k-space trajectories
                nb_ = int(y.shape[0] / nb)
                omega = repeat(
                    self._omega, 'other n_nufft_dims nk -> (other other_rep) n_nufft_dims nk', other=nb, other_rep=nb_
                )
            else:
                omega = self._omega
            y = y.contiguous() if y.stride()[-1] != 1 else y
            y = self._adj_nufft_op(y, omega, norm='ortho')

            # get back to orginal image shape
            nz, ny, nx = self._recon_shape.z, self._recon_shape.y, self._recon_shape.x
            y = rearrange(y, self._target_pattern + '->' + init_pattern, other=nb, coils=nc, dim2=nz, dim1=ny, dim0=nx)

        return y
