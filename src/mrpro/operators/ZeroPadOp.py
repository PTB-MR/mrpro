"""Class for Zero Pad Operator."""

from collections.abc import Sequence

import torch

from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils import zero_pad_or_crop


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
        """Pad or crop data.

        Parameters
        ----------
        x
            data with shape orig_shape

        Returns
        -------
            data with shape padded_shape
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply ZeroPadOp.

        Use `operator.__call__`, i.e. call `operator()` instead.
        """
        return (zero_pad_or_crop(x, self.padded_shape, self.dim),)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operation which crops the padded data back to its original shape.

        Parameters
        ----------
        y
            tensor with padded dimensions

        Returns
        -------
            tensor with cropped dimensions
        """
        return (zero_pad_or_crop(y, self.original_shape, self.dim),)
