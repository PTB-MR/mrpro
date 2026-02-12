"""Zero Operator."""

import torch

from mr2.operators.LinearOperator import LinearOperator


class ZeroOp(LinearOperator):
    """A constant zero operator.

    This operator always returns zero when applied to a tensor.
    It is the neutral element of the addition of operators.
    """

    def __init__(self, keep_shape: bool = False):
        """Initialize the Zero Operator.

        Returns a constant zero, either as a scalar or as a tensor of the same shape as the input,
        depending on the value of keep_shape.
        Returning a scalar can save memory and computation time in some cases.

        Parameters
        ----------
        keep_shape
            If `True`, the shape of the input is kept.
            If `False`, the output is, regardless of the input shape, an integer scalar 0,
            which can broadcast to the input shape and dtype.
        """
        self.keep_shape = keep_shape
        super().__init__()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the zero operator to the input tensor.

        This operator returns a tensor of zeros. Depending on the `keep_shape`
        attribute set during initialization, the output will either be a
        tensor of zeros with the same shape as the input `x`, or a scalar zero.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            A tensor of zeros. This will be `torch.zeros_like(x)`
            if `keep_shape` is `True`, or `torch.tensor(0)` if `keep_shape` is `False`.

        .. note::
            Prefer calling the instance of the ZeroOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of ZeroOp.

        .. note::
            Prefer calling the instance of the ZeroOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if self.keep_shape:
            return (torch.zeros_like(x),)
        else:
            return (torch.tensor(0),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the zero operator.

        Since the zero operator is self-adjoint (mapping everything to zero),
        this method behaves identically to the forward operation. It returns
        a tensor of zeros, either with the same shape as input `x` or as a
        scalar zero, depending on `keep_shape`.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            A tensor of zeros. This will be `torch.zeros_like(x)`
            if `keep_shape` is `True`, or `torch.tensor(0)` if `keep_shape` is `False`.

        """
        if self.keep_shape:
            return (torch.zeros_like(x),)
        else:
            return (torch.tensor(0),)

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        """Adjoint of the Zero Operator."""
        return self
