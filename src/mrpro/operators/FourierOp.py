"""Fourier Operator."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import torch
from torchkbnufft import KbNufft, KbNufftAdjoint

from mrpro.data._kdata.KData import KData
from mrpro.data.enums import TrajType
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator


class FourierOp(LinearOperator):
    """Fourier Operator class."""

    def __init__(
        self,
        recon_matrix: SpatialDimension[int],
        encoding_matrix: SpatialDimension[int],
        traj: KTrajectory,
        force_nufft: bool = False,
        nufft_factory:None=None
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
        force_nufft
            force the use of the NUFFT operator instead of the FFT operator
            This will be slower but can be useful for testing or if gradients
            with respect to the trajectory are needed.
        nufft_factory
            factory function to create the NUFFT operators. If None, a default
            torchkbnufft will be used.
        """
        super().__init__()


        def get_traj(traj: KTrajectory, dims: Sequence[int]):
            return [k for k, i in zip((traj.kz, traj.ky, traj.kx), (-3, -2, -1), strict=True) if i in dims]

        self._ignore_dims, self._fft_dims_k2k1k0, self._nufft_dims = [], [], []
        type_matrix_zyx_210 = traj.type_matrix



        for kzyx, dim_type in zip((-3, -2, -1), traj.type_along_kzyx, strict=True):
            if dim_type & TrajType.SINGLEVALUE:
                # dimension which do not require any transform
                self._ignore_dims.append(kzyx)
            elif dim_type & TrajType.ONGRID and not force_nufft:

                self._fft_dims_k2k1k0.append(kzyx)
            else:
                self._nufft_dims.append(kzyx)

        if self._fft_dims_k2k1k0:
            self._fast_fourier_op = FastFourierOp(
                dim=tuple(self._fft_dims_k2k1k0),
                recon_matrix=[recon_matrix.zyx[d] for d in self._fft_dims_k2k1k0],
                encoding_matrix=[encoding_matrix.zyx[d] for d in self._fft_dims_k2k1k0],
            )

        # Find dimensions which require NUFFT
        if self._nufft_dims:
            fft_dims_k210 = [
                dim
                for dim in (-3, -2, -1)
                if (traj.type_along_k210[dim] & TrajType.ONGRID)
                and not (traj.type_along_k210[dim] & TrajType.SINGLEVALUE)
            ]
            if self._fft_dims_k2k1k0 != fft_dims_k210:
                raise NotImplementedError(
                    'If both FFT and NUFFT dims are present, Cartesian FFT dims need to be aligned with the '
                    'k-space dimension, i.e. kx along k0, ky along k1 and kz along k2',
                )

            self._nufft_im_size = [recon_matrix.zyx[d] for d in self._nufft_dims]
            grid_size = [int(size * nufft_oversampling) for size in self._nufft_im_size]
            omega = [
                k * 2 * torch.pi / ks
                for k, ks in zip(
                    get_traj(traj, self._nufft_dims),
                    [encoding_matrix.zyx[d] for d in self._nufft_dims],
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

    @classmethod
    def from_kdata(cls, kdata: KData, recon_shape: SpatialDimension[int] | None = None) -> Self:
        """Create an instance of FourierOp from kdata with default settings.

        Parameters
        ----------
        kdata
            k-space data
        recon_shape
            dimension of the reconstructed image. Defaults to KData.header.recon_matrix
        """
        return cls(
            recon_matrix=kdata.header.recon_matrix if recon_shape is None else recon_shape,
            encoding_matrix=kdata.header.encoding_matrix,
            traj=kdata.traj,
        )

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
        if len(self._fft_dims_k2k1k0):
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
        if self._fft_dims_k2k1k0:
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
