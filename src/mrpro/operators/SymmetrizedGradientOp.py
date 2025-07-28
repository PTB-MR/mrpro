"""Class for Symmetrized Gradient Operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.LinearOperatorMatrix import LinearOperatorMatrix


class SymmetrizedGradientOp(LinearOperator):
    """
    Symmetrized Gradient Operator.

    TODO: Add documentation for proof of the adjoint property.
    Write the formulas for the 2D case from the
    "Recovering Piecewise Smooth Multichannel Images by Minimization of Convex Functionals with Total Generalized
    Variation Penalty"
    paper. Then write the 3D and proof for the general d-dimensional case.
    """

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'central',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ) -> None:
        """Symmetrized gradient operator.

        Parameters
        ----------
        dim
            Dimension along which finite differences are calculated.
            k elements for k gradients.
        mode
            Type of finite difference operator
        pad_mode
            Padding to ensure output has the same size as the input
        """
        super().__init__()
        self.finite_difference_op = FiniteDifferenceOp(dim, mode=mode, pad_mode=pad_mode)
        adjoint_mode: Literal['central', 'forward', 'backward'] = (
            'backward' if mode == 'forward' else 'forward' if mode == 'backward' else 'central'
        )
        # 1xk matrix
        self.adjoint_finite_difference_op_matrix = LinearOperatorMatrix(
            [[FiniteDifferenceOp([d], mode=adjoint_mode, pad_mode=pad_mode) for d in dim]]
        )

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward of symmetrized gradient.

        Parameters
        ----------
        v
            Input tensor

        Returns
        -------
            Symmetrized gradient of v
        """
        jacobian: torch.Tensor = self.finite_difference_op.forward(v)[0]
        symmetrized_gradient: torch.Tensor = (jacobian + jacobian.transpose(0, 1)) / 2
        return (symmetrized_gradient,)

    def adjoint(self, w: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of symmetrized gradient.

        Parameters
        ----------
        w
            Input tensor

        Returns
        -------
            Adjoint of symmetrized gradient of w
        """
        rows = torch.unbind(w, dim=0)
        return (-self.adjoint_finite_difference_op_matrix.forward(*rows)[0][0],)
