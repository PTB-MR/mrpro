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
from torchkbnufft import KbNufft
from torchkbnufft import KbNufftAdjoint

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.operators import LinearOperator
from mrpro.utils import fft


def is_on_cartesian_grid(x: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """Check if a tensor contains sampling points on a Cartesian grid.

    Parameters
    ----------
    x
        tensor to be checked
    atol
        tolerance of how close the values have to be to the grid points

    Returns
    -------
        boolean tensor if sampling points are on a Cartesian grid.

    If the tensor has batch-dimension>1, this is checked for all batches.
    """

    if x.shape[1:].count(1) <= 1:
        raise ValueError('x is allowed to have at most one non-singleton dimension')
    all_close_list = [
        torch.allclose(x[kb, ...].to(dtype=torch.int64).to(dtype=x.dtype), x[kb, ...], atol=atol)
        for kb in range(x.shape[0])
    ]

    return torch.tensor(all_close_list).all()


class FourierOp(LinearOperator):
    def __init__(
        self,
        recon_shape: SpatialDimension[int],
        encoding_shape: SpatialDimension[int],
        traj: KTrajectory,
        oversampling: SpatialDimension[float] = SpatialDimension(2.0, 2.0, 2.0),
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

        grid_size = []
        nufft_im_size = []
        nufft_dims = []
        fft_dims = []
        ignore_dims = []
        fft_recon_shape = []
        fft_encoding_shape = []
        omega = []
        traj_shape = []
        super().__init__()

        # create information about image shape, k-data shape etc
        # and identify which directions can be ignored, which ones require a nuFFT
        # and for which ones a simple FFT suffices
        for rs, es, os, k, i in zip(
            (recon_shape.z, recon_shape.y, recon_shape.x),
            (encoding_shape.z, encoding_shape.y, encoding_shape.x),
            (oversampling.z, oversampling.y, oversampling.x),
            (traj.kz, traj.ky, traj.kx),
            (-3, -2, -1),
        ):
            nk_list = [traj.kz.shape[i], traj.ky.shape[i], traj.kx.shape[i]]
            if rs <= 1 and nk_list.count(1) == 3:
                # dimension with no Fourier transform
                ignore_dims.append(i)

            elif nk_list.count(1) == 2:  # and is_uniform(k): #TODO: maybe is_uniform never needed?
                # append dimension and output shape for oversampled FFT
                fft_dims.append(i)
                fft_recon_shape.append(int(rs))
                fft_encoding_shape.append(int(es))

            else:
                # dimension with nuFFT
                grid_size.append(int(os * rs))
                nufft_im_size.append(rs)
                nufft_dims.append(i)

                # TODO: can omega be created here already?
                # seems necessary to do this in a different loop where we create a list
                # of kz, ky and kx to be stacked if they are compatible
                # omega.append(k.flatten(start_dim=-3))

            traj_shape.append(k.shape)

        # if uniform directions were identified, create index to perform FFT
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

        # if non-uniform directions were identified, create the trajectories to perform the nuFFT
        if len(nufft_dims) > 0:
            for k, ks in zip((traj.kz, traj.ky, traj.kx), (encoding_shape.z, encoding_shape.y, encoding_shape.x)):
                # check number of singleton dimensions
                n_singleton_dims = k.shape[1:].count(1)
                if n_singleton_dims <= 1:
                    # if not is_uniform(k): #TODO: is_uniform maybe never needed?
                    # Scale trajectory to [-pi pi] required by torchkbnufft
                    omega.append(k.flatten(start_dim=-3) * 2 * torch.pi / ks)

            self._omega = torch.stack(omega, dim=-2)
            self._fwd_nufft_op = KbNufft(
                im_size=nufft_im_size, grid_size=grid_size, numpoints=numpoints, kbwidth=kbwidth
            )
            self._adj_nufft_op = KbNufftAdjoint(
                im_size=nufft_im_size, grid_size=grid_size, numpoints=numpoints, kbwidth=kbwidth
            )

        self._ignore_dims = tuple(ignore_dims)
        self._nufft_dims = tuple(nufft_dims)
        self._fft_dims = tuple(fft_dims)
        self._fft_recon_shape = tuple(fft_recon_shape)
        self._fft_encoding_shape = tuple(fft_encoding_shape)
        self._kshape = torch.broadcast_shapes(*traj_shape)
        self._recon_shape = recon_shape
        self._encoding_shape = encoding_shape
        self._nufft_im_size = nufft_im_size

    @staticmethod
    def get_target_pattern(fft_dims: tuple[int, ...], nufft_dims: tuple[int, ...], ignore_dims: tuple[int, ...]) -> str:
        """Pattern to reshape image/k-space data to be able to perform nuFFT.

        Parameters
        ----------
        fft_dims
            the dimensions along which a simple FFT can be performed
        nufft_dims
            the dimensions for which a nuFFT has to be performed
        ignore_dims
            dimension that are not Fourier-transformed

        Returns
        -------
            pattern for how to reshape an image (other coils z y x) / k-space data (other coils k2 k1 k0)
            to be able to apply the forward/adoint nuFFT.
        """
        fft_dims_str = ''
        nufft_dims_str = ''
        ignore_dims_str = ''

        for dim, dim_str in zip((-3, -2, -1), ('dim2', 'dim1', 'dim0')):
            if dim in fft_dims:
                fft_dims_str += ' ' + dim_str
            elif dim in nufft_dims:
                nufft_dims_str += ' ' + dim_str
            elif dim in ignore_dims:
                ignore_dims_str += ' ' + dim_str

        target_pattern = f'(other{fft_dims_str}{ignore_dims_str}) coils{nufft_dims_str}'
        return target_pattern

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

        if len(self._nufft_dims) != 0:
            init_pattern = 'other coils dim2 dim1 dim0'
            target_pattern = self.get_target_pattern(self._fft_dims, self._nufft_dims, self._ignore_dims)

            # get shape before applying nuFFT
            nb, nc, _, _, _ = x.shape
            x = rearrange(x, init_pattern + '->' + target_pattern)

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
            if -1 in self._fft_dims:
                nk0 = self._recon_shape.x
            if -2 in self._fft_dims:
                nk1 = self._recon_shape.y
            if -3 in self._fft_dims:
                nk2 = self._recon_shape.z
            x = rearrange(x, target_pattern + '->' + init_pattern, other=nb, coils=nc, dim1=nk1, dim2=nk2, dim0=nk0)

        if len(self._fft_dims) != 0:
            xfft = fft.image_to_kspace(x, encoding_shape=self._fft_encoding_shape, dim=self._fft_dims)

            if len(self._fft_idx_full) == 0 or self._fft_idx_full.shape[1] != x.shape[1]:
                coil_idx = torch.ones(
                    *(x.shape[:2] + self._fft_idx.shape[2:]), dtype=torch.int64, device=x.device
                ) * rearrange(
                    torch.linspace(0, x.shape[1] - 1, x.shape[1], dtype=torch.int64, device=x.device),
                    'coils->1 coils 1 1 1',
                )
                other_idx = torch.ones_like(self._fft_idx, dtype=torch.int64, device=x.device) * rearrange(
                    torch.linspace(0, x.shape[0] - 1, x.shape[0], dtype=torch.int64, device=x.device),
                    'other->other 1 1 1 1',
                )
                self._fft_idx_full = (
                    self._fft_idx
                    + other_idx * x.shape[1]
                    + coil_idx * self._encoding_shape.z * self._encoding_shape.y * self._encoding_shape.x
                )

            x = torch.take(xfft, self._fft_idx_full)

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

        if self._kshape[1:] != y.shape[-3:]:
            raise ValueError('k-space data shape missmatch')

        # apply IFFT
        if len(self._fft_dims) != 0:
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

            yfft = torch.zeros(
                *(y.shape[:2] + (self._encoding_shape.z, self._encoding_shape.y, self._encoding_shape.x)),
                dtype=y.dtype,
                device=y.device,
            )
            y = fft.kspace_to_image(
                yfft.put_(self._fft_idx_full, y, accumulate=True), recon_shape=self._fft_recon_shape, dim=self._fft_dims
            )

        # move dim where FFT was already performed such nuFFT can be performed
        if len(self._nufft_dims) != 0:
            init_pattern = 'other coils dim2 dim1 dim0'
            target_pattern = self.get_target_pattern(self._fft_dims, self._nufft_dims, self._ignore_dims)

            # get shape before applying nuFFT
            nb, nc, _, _, _ = y.shape
            y = rearrange(y, init_pattern + '->' + target_pattern)

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
            y = rearrange(y, target_pattern + '->' + init_pattern, other=nb, coils=nc, dim2=nz, dim1=ny, dim0=nx)

        return y
