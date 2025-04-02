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
        """Identity of input.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            the input tensor
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply IdentityOp.

        Use `operator.__call__`, i.e. call `operator()` instead.
        """
        return (x,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply adjoint of IdentityOp to the input tensor.

        Parameters
        ----------
        y
            input tensor

        Returns
        -------
            output tensor, identical to input tensor
        """
        return (y,)
