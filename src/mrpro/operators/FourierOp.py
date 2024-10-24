"""Fourier Operator."""

from collections.abc import Sequence

import torch
from typing_extensions import Self

from mrpro.data._kdata.KData import KData
from mrpro.data.enums import TrajType
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.IdentityOp import IdentityOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.NonUniformFastFourierOp import NonUniformFastFourierOp


class FourierOp(LinearOperator):
    """Fourier Operator class."""

    def __init__(
        self,
        recon_matrix: SpatialDimension[int],
        encoding_matrix: SpatialDimension[int],
        traj: KTrajectory,
        nufft_oversampling: float = 2.0,
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
            oversampling used for interpolation in non-uniform FFTs. The oversampling of the interpolation grid, which
            is needed during the non-uniform FFT, ensures that there is no foldover due to the finite gridding kernel.
            It can be reduced (e.g. to 1.25) to speed up the non-uniform FFT but this might lead to poorer image
            quality.
        """
        super().__init__()

        def get_spatial_dims(spatial_dims: SpatialDimension, dims: Sequence[int]):
            return [
                s
                for s, i in zip((spatial_dims.z, spatial_dims.y, spatial_dims.x), (-3, -2, -1), strict=True)
                if i in dims
            ]

        self._ignore_dims, self._fft_dims, self._nufft_dims = [], [], []
        for dim, type_ in zip((-3, -2, -1), traj.type_along_kzyx, strict=True):
            if type_ & TrajType.SINGLEVALUE:
                # dimension which do not require any transform
                self._ignore_dims.append(dim)
            elif type_ & TrajType.ONGRID:
                self._fft_dims.append(dim)
            else:
                self._nufft_dims.append(dim)

        self._fast_fourier_op = (
            FastFourierOp(
                dim=tuple(self._fft_dims),
                recon_matrix=get_spatial_dims(recon_matrix, self._fft_dims),
                encoding_matrix=get_spatial_dims(encoding_matrix, self._fft_dims),
            )
            if self._fft_dims
            else IdentityOp()
        )

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

        self._non_uniform_fast_fourier_op = (
            NonUniformFastFourierOp(
                dim=tuple(self._nufft_dims),
                recon_matrix=get_spatial_dims(recon_matrix, self._nufft_dims),
                encoding_matrix=get_spatial_dims(encoding_matrix, self._nufft_dims),
                traj=traj,
                nufft_oversampling=nufft_oversampling,
            )
            if self._nufft_dims
            else IdentityOp()
        )

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
        # FFT followed by NUFFT
        return self._non_uniform_fast_fourier_op(self._fast_fourier_op(x)[0])

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
        # NUFFT followed by FFT
        return self._fast_fourier_op.adjoint(self._non_uniform_fast_fourier_op.adjoint(x)[0])
