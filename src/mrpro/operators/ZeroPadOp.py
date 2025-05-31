"""Class for Zero Pad Operator."""

from collections.abc import Sequence

import torch

from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils import pad_or_crop


class ZeroPadOp(LinearOperator):
    """Zero Pad operator class."""

    def __init__(self, dim: Sequence[int], original_shape: Sequence[int], padded_shape: Sequence[int]) -> None:
        """Zero Pad Operator class.

        The operator carries out zero-padding if the `padded_shape` is larger than `orig_shape` and cropping if the
        `padded_shape` is smaller.

        Parameters
        ----------
        dim
            dimensions along which padding should be applied
        original_shape
            shape of original data along dim, same length as `dim`
        padded_shape
            shape of padded data along dim, same length as `dim`
        """
        if len(dim) != len(original_shape) or len(dim) != len(padded_shape):
            raise ValueError('Dim, orig_shape and padded_shape have to be of same length')

        super().__init__()
        self.dim = dim
        self.original_shape = original_shape
        self.padded_shape = padded_shape

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply zero-padding or cropping to the input tensor.

        If `padded_shape` (defined at initialization) is larger than `original_shape`
        along the specified dimensions, the tensor is zero-padded. If smaller,
        it is cropped. The operation is applied along the dimensions specified in `dim`.

        Parameters
        ----------
        x
            Input tensor. Its shape along the dimensions in `dim` should
            match `original_shape`.

        Returns
        -------
        tuple[torch.Tensor,]
            The padded or cropped tensor. Its shape along the dimensions in `dim`
            will match `padded_shape`.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of ZeroPadOp.

        Note: Do not use. Instead, call the instance of the Operator as operator(x)"""
        return (pad_or_crop(x, self.padded_shape, self.dim),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply cropping or zero-padding (adjoint of forward).

        This operation is the adjoint of the forward `ZeroPadOp`. If the forward
        operation was padding, the adjoint is cropping. If the forward was
        cropping, the adjoint is zero-padding. The operation is applied along
        the dimensions specified in `dim`.

        Parameters
        ----------
        x
            Input tensor. Its shape along the dimensions in `dim` should
            match `padded_shape` (from initialization).

        Returns
        -------
        tuple[torch.Tensor,]
            The cropped or padded tensor. Its shape along the dimensions in `dim`
            will match `original_shape` (from initialization).
        """
        return (pad_or_crop(x, self.original_shape, self.dim),)
