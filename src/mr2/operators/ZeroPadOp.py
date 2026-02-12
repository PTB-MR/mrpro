"""Class for Zero Pad Operator."""

from collections.abc import Sequence

import torch

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils.pad_or_crop import pad_or_crop


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
            The padded or cropped tensor. Its shape along the dimensions in `dim`
            will match `padded_shape`.


        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of ZeroPadOp.

        .. note::
            Prefer calling the instance of the ZeroPadOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
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
            The cropped or padded tensor. Its shape along the dimensions in `dim`
            will match `original_shape` (from initialization).
        """
        return (pad_or_crop(x, self.original_shape, self.dim),)
