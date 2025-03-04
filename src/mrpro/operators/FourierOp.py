"""Fourier Operator."""

from collections.abc import Sequence

import torch
from typing_extensions import Self

from mrpro.data.enums import TrajType
from mrpro.data.KData import KData
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.CartesianSamplingOp import CartesianSamplingOp
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.NonUniformFastFourierOp import NonUniformFastFourierOp


class FourierOp(LinearOperator, adjoint_as_backward=True):
    """Fourier Operator class.

    This is the recommended operator for all Fourier transformations.
    It auto-detects if a non-uniform or regular fast Fourier transformation is required.
    For Cartesian data on a regular grid, the data is sorted and a FFT is used.
    For non-Cartesian data, a NUFFT with regridding is used.
    It also includes padding/cropping to the reconstruction matrix size.

    The operator can directly be constructed from a `~mrpro.data.KData` object to match its
    trajectory and header information, see `FourierOp.from_kdata`.

    """

    def __init__(
        self,
        recon_matrix: SpatialDimension[int],
        encoding_matrix: SpatialDimension[int],
        traj: KTrajectory,
    ) -> None:
        """Initialize Fourier Operator.

        Parameters
        ----------
        recon_matrix
            dimension of the reconstructed image
        encoding_matrix
            dimension of the encoded k-space
        traj
            the k-space trajectories where the frequencies are sampled
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

        if self._fft_dims:
            self._fast_fourier_op: FastFourierOp | None = FastFourierOp(
                dim=tuple(self._fft_dims),
                recon_matrix=get_spatial_dims(recon_matrix, self._fft_dims),
                encoding_matrix=get_spatial_dims(encoding_matrix, self._fft_dims),
            )
            self._cart_sampling_op: CartesianSamplingOp | None = CartesianSamplingOp(
                encoding_matrix=encoding_matrix, traj=traj
            )
        else:
            self._fast_fourier_op = None
            self._cart_sampling_op = None

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
                    'k-space dimension, i.e. kx along k0, ky along k1 and kz along k2.',
                )

            self._non_uniform_fast_fourier_op: NonUniformFastFourierOp | None = NonUniformFastFourierOp(
                direction=tuple(self._nufft_dims),  # type: ignore[arg-type]
                recon_matrix=get_spatial_dims(recon_matrix, self._nufft_dims),
                encoding_matrix=get_spatial_dims(encoding_matrix, self._nufft_dims),
                traj=traj,
            )
        else:
            self._non_uniform_fast_fourier_op = None

        self._trajectory_info = repr(traj)

    @classmethod
    def from_kdata(cls, kdata: KData, recon_shape: SpatialDimension[int] | None = None) -> Self:
        """Create an instance of FourierOp from kdata with default settings.

        Parameters
        ----------
        kdata
            k-space data
        recon_shape
            Dimension of the reconstructed image. Defaults to `KData.header.recon_matrix`.
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
            coil image data with shape: `(... coils z y x)`

        Returns
        -------
            coil k-space data with shape: `(... coils k2 k1 k0)`
        """
        # NUFFT Type 2 followed by FFT
        if self._non_uniform_fast_fourier_op:
            (x,) = self._non_uniform_fast_fourier_op(x)

        if self._fast_fourier_op and self._cart_sampling_op:
            (x,) = self._cart_sampling_op(self._fast_fourier_op(x)[0])
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator mapping the coil k-space data to the coil images.

        Parameters
        ----------
        x
            coil k-space data with shape: `(... coils k2 k1 k0)`

        Returns
        -------
            coil image data with shape: `(... coils z y x)`
        """
        # FFT followed by NUFFT Type 1
        if self._fast_fourier_op and self._cart_sampling_op:
            (x,) = self._fast_fourier_op.adjoint(self._cart_sampling_op.adjoint(x)[0])

        if self._non_uniform_fast_fourier_op:
            (x,) = self._non_uniform_fast_fourier_op.adjoint(x)
        return (x,)

    @property
    def gram(self) -> LinearOperator:
        """Return the gram operator."""
        return FourierGramOp(self)

    def __repr__(self) -> str:
        """Representation method for Fourier Operator."""
        string = ''
        device_omega = None
        device_cart = None

        if self._nufft_dims and self._non_uniform_fast_fourier_op:
            nufftop = self._non_uniform_fast_fourier_op
            string += f'\n{nufftop}\n'
            device_omega = nufftop._omega.device if nufftop._omega.device is not None else None
        if self._fft_dims:
            string += f'\n{self._fast_fourier_op}\n\n{self._cart_sampling_op}\n'
            device_cart = self._cart_sampling_op._fft_idx.device if self._cart_sampling_op is not None else None

        if device_omega and device_cart:
            device = device_omega if device_omega == device_cart else 'Different devices'
        else:
            device = device_omega or device_cart or 'None'

        out = f'{type(self).__name__} on device: {device}\n{string}\n{self._trajectory_info}'
        return out


class FourierGramOp(LinearOperator):
    """Gram operator for the Fourier operator.

    Implements the adjoint of the forward operator of the Fourier operator, i.e. the gram operator
    `F.H@F`.

    Uses a convolution, implemented as multiplication in Fourier space, to calculate the gram operator
    for the toeplitz NUFFT operator.

    Uses a multiplication with a binary mask in Fourier space to calculate the gram operator for
    the Cartesian FFT operator

    This Operator is only used internally and should not be used directly.
    Instead, consider using the py:func:`~FourierOp.gram` property of py:class:`FourierOp`.
    """

    _kernel: torch.Tensor | None

    def __init__(self, fourier_op: FourierOp) -> None:
        """Initialize the gram operator.

        If density compensation weights are provided, they the operator
        F.H@dcf@F is calculated.

        Parameters
        ----------
        fourier_op
            the Fourier operator to calculate the gram operator for

        """
        super().__init__()
        if fourier_op._non_uniform_fast_fourier_op:
            self.nufft_gram: None | LinearOperator = fourier_op._non_uniform_fast_fourier_op.gram
        else:
            self.nufft_gram = None

        if fourier_op._fast_fourier_op and fourier_op._cart_sampling_op:
            self.fast_fourier_gram: None | LinearOperator = (
                fourier_op._fast_fourier_op.H @ fourier_op._cart_sampling_op.gram @ fourier_op._fast_fourier_op
            )
        else:
            self.fast_fourier_gram = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape: `(..., coils, z, y, x)`
        """
        if self.nufft_gram:
            (x,) = self.nufft_gram(x)

        if self.fast_fourier_gram:
            (x,) = self.fast_fourier_gram(x)
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape: `(..., coils, k2, k1, k0)`
        """
        return self.forward(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the gram operator."""
        return self
