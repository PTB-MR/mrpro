"""Zero Operator."""

import torch

from mrpro.operators.LinearOperator import LinearOperator


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
            If True, the shape of the input is kept.
            If False, the output is, regardless of the input shape, an integer scalar 0,
            which can broadcast to the input shape and dtype.
        """
        self.keep_shape = keep_shape
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the operator to the input.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        zeros_like(x) or scalar 0
        """
        if self.keep_shape:
            return (torch.zeros_like(x),)
        else:
            return (torch.tensor(0),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the operator to the input.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        zeros_like(x)
        """
        if self.keep_shape:
            return (torch.zeros_like(x),)
        else:
            return (torch.tensor(0),)
