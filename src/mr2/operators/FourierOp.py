"""Fourier Operator."""

from collections.abc import Sequence
from functools import cached_property

import torch
from typing_extensions import Self

from mr2.data.enums import TrajType
from mr2.data.KData import KData
from mr2.data.KTrajectory import KTrajectory
from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.CartesianSamplingOp import CartesianSamplingOp
from mr2.operators.FastFourierOp import FastFourierOp
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.NonUniformFastFourierOp import NonUniformFastFourierOp


class FourierOp(LinearOperator, adjoint_as_backward=True):
    """Fourier Operator class.

    This is the recommended operator for all Fourier transformations.
    It auto-detects if a non-uniform or regular fast Fourier transformation is required.
    For Cartesian data on a regular grid, the data is sorted and a FFT is used.
    For non-Cartesian data, a NUFFT with regridding is used.
    It also includes padding/cropping to the reconstruction matrix size.

    The operator can directly be constructed from a `~mr2.data.KData` object to match its
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

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward Fourier operation (image to k-space).

        This operator maps coil image data to coil k-space data.
        Depending on the trajectory and dimensions, it may involve NUFFT, FFT,
        and Cartesian sampling operations.

        Parameters
        ----------
        x
            Coil image data, typically with shape `(... coils z y x)`.

        Returns
        -------
            Coil k-space data, typically with shape `(... coils k2 k1 k0)`.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of FourierOp.

        .. note::
            Prefer calling the instance of the FourierOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        # NUFFT Type 2 followed by FFT
        if self._non_uniform_fast_fourier_op:
            (x,) = self._non_uniform_fast_fourier_op(x)

        if self._fast_fourier_op and self._cart_sampling_op:
            (x,) = self._cart_sampling_op(self._fast_fourier_op(x)[0])
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply adjoint Fourier operation (k-space to image).

        This operator maps coil k-space data to coil image data.
        It is the adjoint of the forward operation and involves corresponding
        adjoint NUFFT, FFT, and Cartesian sampling operations.

        Parameters
        ----------
        x
            Coil k-space data, typically with shape `(... coils k2 k1 k0)`.

        Returns
        -------
            Coil image data, typically with shape `(... coils z y x)`.
        """
        # FFT followed by NUFFT Type 1
        if self._fast_fourier_op and self._cart_sampling_op:
            (x,) = self._fast_fourier_op.adjoint(self._cart_sampling_op.adjoint(x)[0])

        if self._non_uniform_fast_fourier_op:
            (x,) = self._non_uniform_fast_fourier_op.adjoint(x)
        return (x,)

    @cached_property
    def gram(self) -> LinearOperator:
        """Return the gram operator."""
        try:
            return FourierGramOp(self)
        except NotImplementedError:
            return self.H @ self

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

        if fourier_op._fast_fourier_op and fourier_op._cart_sampling_op:
            sampling_gram = fourier_op._cart_sampling_op.gram
            if (
                fourier_op._non_uniform_fast_fourier_op
                and sampling_gram.mask is not None
                and any(sampling_gram.mask.shape[d] != 1 for d in fourier_op._nufft_dims)
            ):
                raise NotImplementedError(
                    'FourierGramOp does not support non-uniform FFTs combined with Cartesian sampling '
                    'that differs along the NUFFT dimensions.'
                )
            self.fast_fourier_gram: None | LinearOperator = (
                fourier_op._fast_fourier_op.H @ fourier_op._cart_sampling_op.gram @ fourier_op._fast_fourier_op
            )
        else:
            self.fast_fourier_gram = None

        if fourier_op._non_uniform_fast_fourier_op:
            self.nufft_gram: None | LinearOperator = fourier_op._non_uniform_fast_fourier_op.gram
        else:
            self.nufft_gram = None

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the Gram operator of the FourierOp (F.H @ F).

        This operation applies the composition of the adjoint Fourier operator
        and the forward Fourier operator. It may involve Gram operators
        of NUFFT and/or FFT components.

        Parameters
        ----------
        x
            Input tensor, typically image-space data with shape `(..., coils, z, y, x)`.

        Returns
        -------
            Output tensor, typically image-space data after F.H @ F has been applied.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of FourierGramOp.

        .. note::
            Prefer calling the instance of the FourierGramOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if self.nufft_gram:
            (x,) = self.nufft_gram(x)

        if self.fast_fourier_gram:
            (x,) = self.fast_fourier_gram(x)
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the Gram operator.

        Since the Gram operator (F.H @ F) is self-adjoint, this method
        calls the forward operation.

        Parameters
        ----------
        x
            Input tensor, typically image-space data with shape `(..., coils, z, y, x)`.
            Note: The original docstring mentioned k-space shape, but for a self-adjoint
            image-to-image Gram operator, the input to adjoint should match input to forward.

        Returns
        -------
            Output tensor, same as the forward operation.
        """
        return super().__call__(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the gram operator."""
        return self
