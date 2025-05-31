"""Identity Operator."""

import torch

from mrpro.operators.LinearOperator import LinearOperator


class IdentityOp(LinearOperator):
    r"""The Identity Operator.

    A Linear Operator that returns a single input unchanged.
    """

    def __init__(self) -> None:
        """Initialize Identity Operator."""
        super().__init__()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the identity operation.

        This operator returns the input tensor unchanged.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The input tensor, unchanged.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of IdentityOp.

.. note::
   Prefer calling the instance of the IdentityOp operator as ``operator(x)`` over directly calling this method.
"""
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the adjoint of the identity operation.

        Since the identity operator is self-adjoint, this method returns
        the input tensor unchanged.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The input tensor, unchanged.
        """
        return (x,)
