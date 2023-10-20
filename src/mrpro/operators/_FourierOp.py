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
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from torchkbnufft import KbNufft
from torchkbnufft import KbNufftAdjoint

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.operators import LinearOperator


def is_uniform(x: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """Check if a tensor contains equidistant sampling points.

    If the tensor has batch-dimension>1, this is checked for all
    batches.
    """
    # TODO: currrently, this function is never used here.

    if x.shape[1:].count(1) <= 1:
        raise ValueError('x is allowed to have at most one non-singleton dimension')
    all_close_list = [
        torch.allclose(torch.diff(x[kb, ...].flatten()), torch.diff(x[kb, ...].flatten())[0], atol=atol)
        for kb in range(x.shape[0])
    ]

    return torch.tensor(all_close_list).all()


class FourierOp(LinearOperator):
    def __init__(
        self,
        im_shape: SpatialDimension[int],
        traj: KTrajectory,
        oversampling: SpatialDimension[float] = SpatialDimension(
            2.0, 2.0, 2.0
        ),  # TODO: maybe not the nicest, since only used for nuFFT
        numpoints: int = 6,
        kbwidth: float = 2.34,
    ):
        """Fourier Operator class.

        Parameters
        ----------
        im_shape
            dimension of the image to be Fourier-transformed

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
        fft_s = []
        omega = []
        traj_shape = []
        super().__init__()

        # create information about image shape, k-data shape etc
        # and identify which directions can be ignored, which ones require a nuFFT
        # and for which ones a simple FFT suiffices
        for n, os, k, i in zip(
            (im_shape.z, im_shape.y, im_shape.x),
            (oversampling.z, oversampling.y, oversampling.x),
            (traj.kz, traj.ky, traj.kx),
            (-3, -2, -1),
        ):
            nk_list = [traj.kz.shape[i], traj.ky.shape[i], traj.kx.shape[i]]
            if n <= 1 and nk_list.count(1) == 3:
                # dimension with no Fourier transform
                ignore_dims.append(i)

            elif nk_list.count(1) == 2:  # and is_uniform(k): #TODO: maybe is_uniform never needed?
                # dimension with FFT
                nk = torch.tensor(nk_list)
                supp = torch.where(nk != 1, nk, 0)
                s = supp.max().item()

                # append dimension and output shape for oversampled FFT
                fft_dims.append(i)
                fft_s.append(s)
            else:
                # dimension with nuFFT
                grid_size.append(int(os * n))
                nufft_im_size.append(n)
                nufft_dims.append(i)

                # TODO: can omega be created here already?
                # seems necessary to do this in a different loop where we create a list
                # of kz, ky and kx to be stacked if they are compatible
                # omega.append(k.flatten(start_dim=-3))

            traj_shape.append(k.shape)

        # if non-uniform directions were identified, create the trajectories
        # to perform the nuFFT
        if len(nufft_dims) != 0:
            for k in (traj.kz, traj.ky, traj.kx):
                # check number of singleton dimensions
                n_singleton_dims = k.shape[1:].count(1)
                if n_singleton_dims <= 1:
                    # if not is_uniform(k): #TODO: is_uniform maybe never needed?
                    omega.append(k.flatten(start_dim=-3))

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
        self._fft_s = tuple(fft_s)
        self._kshape = torch.broadcast_shapes(*traj_shape)
        self._im_shape = im_shape
        self._nufft_im_size = nufft_im_size

    @staticmethod
    def get_target_pattern(fft_dims: tuple, nufft_dims: tuple, ignore_dims: tuple) -> str:
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
            pattern for how to reshape an image (other coil z y x) / k-space data (other coil k2 k1 k0)
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

        target_pattern = f'(other{fft_dims_str}{ignore_dims_str}) coil{nufft_dims_str}'
        return target_pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward operator mapping the coil-images to the coil k-space data.

        Parameters
        ----------
        x
            coil image data with shape: other,coil,z,y,x

        Returns
        -------
            coil k-space data with shape: other,coil,k2,k1,k0
        """
        if self._im_shape != SpatialDimension(*x.shape[-3:]):
            raise ValueError('image data shape missmatch')

        if len(self._fft_dims) != 0:
            x = torch.fft.fftn(x, s=self._fft_s, dim=self._fft_dims, norm='ortho')

        if len(self._nufft_dims) != 0:
            init_pattern = 'other coil dim2 dim1 dim0'
            target_pattern = self.get_target_pattern(self._fft_dims, self._nufft_dims, self._ignore_dims)

            # get shape before applying nuFFT
            nb, nc, nz, ny, nx = x.shape
            x = rearrange(x, init_pattern + '->' + target_pattern)

            # apply nuFFT
            if nb > 1 and len(self._fft_dims) >= 1 and len(self._nufft_dims) >= 1:
                # if multiple batches, repeat k-space trajectories
                nb_ = int(x.shape[0] / nb)
                omega = repeat(
                    self._omega, 'other n_nufft_dims nk -> (other other_rep) n_nufft_dims nk', other_rep=nb_, other=nb
                )
            else:
                omega = self._omega
            x = self._fwd_nufft_op(x, omega, norm='ortho')

            # identify separate dimensionality of nuFFT-dimensions
            nufft_dim_size = tuple([nk for nk, dim in zip(self._kshape[1:], (-3, -2, -1)) if dim in self._nufft_dims])

            # unflatten the nuFFT-dimensions
            x = x.view(*x.shape[:2], *nufft_dim_size)

            # bring to shape defined by k-space trajectories
            nk2, nk1, nk0 = self._kshape[1:]
            x = rearrange(x, target_pattern + '->' + init_pattern, other=nb, coil=nc, dim1=nk1, dim2=nk2, dim0=nk0)

        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint operator mapping the coil k-space data to the coil images.

        Parameters
        ----------
        y
            coil k-space data with shape: other,coil,k2,k1,k0

        Returns
        -------
            coil image data with shape: other,coil,z,y,x
        """

        if self._kshape[1:] != y.shape[-3:]:
            raise ValueError('k-space data shape missmatch')

        # apply IFFT
        if len(self._fft_dims) != 0:
            im_shape = [self._im_shape.z, self._im_shape.y, self._im_shape.x]
            y = torch.fft.ifftn(y, s=self._fft_s, dim=self._fft_dims, norm='ortho')

            # construct the paddings based on the FFT-directions
            # TODO: can this be written more nicely?
            diff_dim = torch.tensor(im_shape) - torch.tensor(y.shape[2:])
            npad_tuple = tuple(
                [
                    int(diff_dim[i // 2].item()) if (i % 2 == 0 and dim in self._fft_dims) else 0
                    for (i, dim) in zip(range(0, 6)[::-1], (-1, -1, -2, -2, -3, -3))
                ]
            )

            # crop using (negative) padding;
            if not torch.all(torch.tensor(npad_tuple) != 0):
                y = F.pad(y, npad_tuple)

        # move dim where FFT was already performed such nuFFT can be performed
        if len(self._nufft_dims) != 0:
            init_pattern = 'other coil dim2 dim1 dim0'
            target_pattern = self.get_target_pattern(self._fft_dims, self._nufft_dims, self._ignore_dims)

            # get shape before applying nuFFT
            nb, nc, _, _, _ = y.shape
            y = rearrange(y, init_pattern + '->' + target_pattern)

            # flatten the nuFFT-dimensions
            nufft_dim_size = tuple([nk for nk, dim in zip(self._kshape[1:], (-3, -2, -1)) if dim in self._nufft_dims])
            y_shape_tuple = tuple(y.shape)
            nk = int(torch.prod(torch.tensor(nufft_dim_size)))
            y = y.view(*y_shape_tuple[:2], nk)

            # apply adjoint nuFFT
            if nb > 1 and len(self._fft_dims) >= 1 and len(self._nufft_dims) >= 1:
                # if multiple batches, repeat k-space trajectories
                nb_ = int(y.shape[0] / nb)
                omega = repeat(
                    self._omega, 'other n_nufft_dims nk -> (other other_rep) n_nufft_dims nk', other_rep=nb_, other=nb
                )
            else:
                omega = self._omega
            y = y.contiguous() if y.stride()[-1] != 1 else y
            y = self._adj_nufft_op(y, omega, norm='ortho')

            # get back to orginal k-space shape
            nz, ny, nx = self._im_shape.z, self._im_shape.y, self._im_shape.x
            y = rearrange(y, target_pattern + '->' + init_pattern, other=nb, coil=nc, dim2=nz, dim1=ny, dim0=nx)

        return y
