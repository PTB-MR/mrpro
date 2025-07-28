"""Convenience functions for the TGV PDHG operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.LinearOperatorMatrix import LinearOperatorMatrix
from mrpro.operators.SymmetrizedGradientOp import SymmetrizedGradientOp


class K1(LinearOperator):
    """K1 operator for the TGV PDHG operator."""

    def __init__(self, acquisition_operator: LinearOperator, v_shape: Sequence[int]):
        """Initialize K1 operator."""
        super().__init__()
        self.acquisition_operator = acquisition_operator
        self.v_shape = v_shape

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator for K1."""
        return self.acquisition_operator.forward(y[0])

    def adjoint(self, s: torch.Tensor) -> tuple[torch.Tensor]:
        """Adjoint operator for K1."""
        return (torch.cat((self.acquisition_operator.adjoint(s)[0].unsqueeze(0), torch.zeros(self.v_shape)), dim=0),)


class K2(LinearOperator):
    """K2 operator for the TGV PDHG operator."""

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'central',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ):
        """Initialize K2 operator."""
        super().__init__()
        self.finite_difference_op = FiniteDifferenceOp(
            dim=dim,  # Don't use negative indices so that index addition works correctly
            mode=mode,
            pad_mode=pad_mode,
        )

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator for K2."""
        # (N1, N2, ... N_d) --> (k, N1, N2, ..., N_d)
        nabla_x = self.finite_difference_op.forward(y[0])[0]
        return (nabla_x - y[1:],)

    def adjoint(self, s: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator for K2."""
        adjoint_v = self.finite_difference_op.adjoint(s)[0]
        return (torch.cat((adjoint_v.unsqueeze(0), -s), dim=0),)


class K3(LinearOperator):
    """K3 operator for the TGV PDHG operator."""

    def __init__(
        self,
        x_shape: Sequence[int],
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'central',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ):
        """Initialize K3 operator."""
        super().__init__()
        self.x_shape = x_shape
        self.symmetrized_gradient_op = SymmetrizedGradientOp(
            dim=dim,
            mode=mode,
            pad_mode=pad_mode,
        )

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator for K3."""
        e_h_v = self.symmetrized_gradient_op.forward(y[1:])[0]
        return (e_h_v,)

    def adjoint(self, s: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator for K3."""
        adjoint_w = self.symmetrized_gradient_op.adjoint(s)[0]
        return (torch.cat((torch.zeros(self.x_shape).unsqueeze(0), adjoint_w), dim=0),)


def make_tgv_op(
    acquisition_operator: LinearOperator,
    x_shape: Sequence[int],
    dim: Sequence[int],
    mode: Literal['central', 'forward', 'backward'] = 'central',
    pad_mode: Literal['zeros', 'circular'] = 'zeros',
) -> LinearOperatorMatrix:
    """
    Create the TGV operator.

    #         Parameters
    #         ----------
    #             acquisition_operator: LinearOperator
    #                 The acquisition operator.
    #             x_shape: Sequence[int]
    #                 The shape of the input image.
    #             dim: Sequence[int]
    #                 The dimensions of the finite difference operator.
    #                 Since we prepend extra dimensions in the gradient and symmetrized
    #                 gradient operators, the values in `dim` should be negative
    #                 integers so that they don't have to be changed when the input is
    #                 reshaped (e.g. `SymmetrizedGradientOp` expects an input with an
    #                 additional dimension compared to `FiniteDifferenceOp`).

    #                 For example:

    #                     # symmetrized_gradient: V --> W,  (2, 1, 256, 256) --> (2, 2, 1, 256, 256);

    #                     # symmetrized_gradient_adjoint: W --> V,  (2, 2, 1, 256, 256) --> (2, 1, 256, 256).
    """
    if any(d >= 0 for d in dim):
        raise ValueError(f'Dimensions must be negative integers, got {dim}')
    v_shape = (len(dim), *x_shape)
    reversed_mode: Literal['central', 'forward', 'backward'] = (
        'forward' if mode == 'backward' else 'backward' if mode == 'forward' else 'central'
    )
    return LinearOperatorMatrix(
        (
            (K1(acquisition_operator=acquisition_operator, v_shape=v_shape),),
            (K2(dim=dim, mode=mode, pad_mode=pad_mode),),
            (K3(x_shape=x_shape, dim=dim, mode=reversed_mode, pad_mode=pad_mode),),
        )
    )
