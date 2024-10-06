"""Zero Operator."""

from typing import overload

import torch

from mrpro.operators.IdentityOp import IdentityOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.Operator import Operator


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

    @overload  # type: ignore[override]
    def __add__(self, other: torch.Tensor) -> IdentityOp: ...
    @overload
    def __add__(self, other: LinearOperator) -> LinearOperator: ...

    @overload
    def __add__(
        self, other: Operator[torch.Tensor, tuple[torch.Tensor]]
    ) -> Operator[torch.Tensor, tuple[torch.Tensor]]: ...

    @overload
    def __add__(self, other: Operator) -> Operator: ...

    def __add__(self, other: Operator | LinearOperator | torch.Tensor) -> Operator | LinearOperator:
        """Addition.

        Addition with a LinearOperator returns the other operator, not a LinearOperatorSum.
        """
        if isinstance(other, torch.Tensor):
            return IdentityOp().__mul__(other)
        else:
            return other
