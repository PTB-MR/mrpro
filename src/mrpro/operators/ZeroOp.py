"""Zero Operator."""

import torch

from mrpro.operators.LinearOperator import LinearOperator


class ZeroOp(LinearOperator):
    """A constant zero operator.

    This operator always returns zero when applied to a tensor.
    It is the neutral element of the addition of operators.
    """

    def __init__(self):
        """Initialize the Zero Operator."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the operator to the input.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        zeros_like(x)
        """
        return (torch.zeros_like(x),)

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
        return (torch.zeros_like(x),)
