"""Class for Fast Fourier Operator."""

from collections.abc import Sequence
from dataclasses import astuple

import torch

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.ZeroPadOp import ZeroPadOp


class FastFourierOp(LinearOperator):
    """Fast Fourier operator class.

    Applies a Fast Fourier Transformation along selected dimensions with cropping/zero-padding
    along these selected dimensions

    The transformation is done with 'ortho' normalization, i.e. the normalization constant is split between
    forward and adjoint [FFT]_.

    Remark regarding the fftshift/ifftshift:
    fftshift shifts the zero-frequency point to the center of the data, ifftshift undoes this operation.
    The input to both forward and ajoint assumes that the zero-frequency is in the center of the data.
    Torch.fft.fftn and torch.fft.ifftn expect the zero-frequency to be the first entry in the tensor.
    Therefore for forward and ajoint first ifftshift needs to be applied, then fftn or ifftn and then ifftshift.

    References
    ----------
    .. [FFT] FFT https://numpy.org/doc/stable/reference/routines.fft.html

    """

    def __init__(
        self,
        dim: Sequence[int] = (-3, -2, -1),
        recon_matrix: SpatialDimension[int] | Sequence[int] | None = None,
        encoding_matrix: SpatialDimension[int] | Sequence[int] | None = None,
    ) -> None:
        """Initialize a Fast Fourier Operator.

        If both recon_matrix and encoding_matrix are set, the operator will perform padding/cropping before and
        after the transforms to match the shape in image space (recon_matrix) and k-shape (encoding_matrix).
        If both are set to None, no padding or cropping will be performed.
        If these are SpatialDimension, the transform dimensions must be within the last three dimensions,
        typically corresponding to the (k2,k1,k0) and (z,y,x) axes of KData and IData, respectively.


        Parameters
        ----------
        dim
            dim along which FFT and IFFT are applied, by default last three dimensions (-3, -2, -1),
            as these correspond to k2, k1, and k0 of KData.
        encoding_matrix
            shape of encoded k-data along the axes in dim. Must be set if recon_matrix is set.
            If encoding_matrix and recon_matrix are None, no padding or cropping will be performed.
            If all values in dim are -3, -2 or -1, this can also be a SpatialDimension describing the
            k-space shape in all 3 dimensions (k2, k1, k0), but only values in the dimensions in dim will be used.
            Otherwise, it should be a Sequence of the same length as dim.
        recon_matrix
            shape of reconstructed image data. Must be set if encoding_matrix is set.
            If encoding_matrix and recon_matrix are None, no padding or cropping will be performed.
            If all values in dim are -3, -2 or -1, this can also be a SpatialDimension describing the
            image-space shape in all 3 dimensions (z,y,x), but only values in the dimensions in dim will be used.
            Otherwise, it should be a Sequence of the same length as dim.
        """
        super().__init__()
        self._dim = tuple(dim)
        self._pad_op: ZeroPadOp

        if isinstance(recon_matrix, SpatialDimension):
            if not all(d in (-1, -2, -3) for d in dim):
                raise NotImplementedError(
                    f'recon_matrix can only be a SpatialDimension if each value in dim is in (-3,-2,-1),'
                    f'got {dim=}\nInstead, you can also supply a list of values of same length as dim'
                )
            original_shape: Sequence[int] | None = [int(astuple(recon_matrix)[d]) for d in dim]

        else:
            original_shape = recon_matrix

        if isinstance(encoding_matrix, SpatialDimension):
            if not all(d in (-1, -2, -3) for d in dim):
                raise NotImplementedError(
                    f'encoding_matrix can only be a SpatialDimension if each value in dim is in (-3,-2,-1),'
                    f'got {dim=}\nInstead, you can also supply a list of values of same length as dim'
                )
            padded_shape: Sequence[int] | None = [int(astuple(encoding_matrix)[d]) for d in dim]

        else:
            padded_shape = encoding_matrix

        if original_shape is not None and padded_shape is not None:
            # perform padding / cropping
            self._pad_op = ZeroPadOp(dim=dim, original_shape=original_shape, padded_shape=padded_shape)
        elif encoding_matrix is None and recon_matrix is None:
            # No padding no padding / cropping
            self._pad_op = ZeroPadOp(dim=(), original_shape=(), padded_shape=())
        else:
            raise ValueError(
                'Either encoding_matrix and recon_matrix must both be set to None or both to a value, got'
                f'{encoding_matrix=} and {recon_matrix=}'
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """FFT from image space to k-space.

        Parameters
        ----------
        x
            image data on Cartesian grid

        Returns
        -------
            FFT of x
        """
        y = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(*self._pad_op(x), dim=self._dim), dim=self._dim, norm='ortho'),
            dim=self._dim,
        )
        return (y,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """IFFT from k-space to image space.

        Parameters
        ----------
        y
            k-space data on Cartesian grid

        Returns
        -------
            IFFT of y
        """
        # FFT
        return self._pad_op.adjoint(
            torch.fft.fftshift(
                torch.fft.ifftn(torch.fft.ifftshift(y, dim=self._dim), dim=self._dim, norm='ortho'),
                dim=self._dim,
            ),
        )
