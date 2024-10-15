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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Identity of input.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            the input tensor
        """
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Adjoint Identity.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            the input tensor
        """
        return (x,)
