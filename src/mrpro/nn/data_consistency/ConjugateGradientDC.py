"""Conjugate gradient data consistency."""

import torch
from torch.nn import Module, Parameter

from mrpro.data.CsmData import CsmData
from mrpro.data.KData import KData
from mrpro.operators.ConjugateGradientOp import ConjugateGradientOp
from mrpro.operators.FourierOp import FourierOp
from mrpro.operators.IdentityOp import IdentityOp
from mrpro.operators.SensitivityOp import SensitivityOp


class ConjugateGradientDC(Module):
    """Conjugate gradient data consistency."""

    def __init__(self, initial_regularization_weight: torch.Tensor | float):
        """Initialize the conjugate gradient data consistency.

        Parameters
        ----------
        initial_regularization_weight
            Initial regularization weight.
        """
        super().__init__()
        self.regularization_weight = Parameter(torch.as_tensor(initial_regularization_weight))

        def operator_factory(
            fourier_op: FourierOp, csm: torch.Tensor | CsmData | None, regularization_weight: torch.Tensor | float, *_
        ):
            csm_op = SensitivityOp(csm) if csm is not None else IdentityOp()
            op = csm_op.H @ fourier_op.gram @ csm_op + regularization_weight
            return op

        self.cg_op = ConjugateGradientOp(
            operator_factory=operator_factory,
            rhs_factory=lambda _fourier, _csm, regularization_weight, zero_filled, regularization: zero_filled
            + regularization_weight * regularization,
        )

    def forward(
        self,
        x: torch.Tensor,
        data: torch.Tensor | KData,
        fourier_op: FourierOp,
        csm: torch.Tensor | CsmData | None,
    ):
        data_ = data.data if isinstance(data, KData) else data
        zero_filled = fourier_op.adjoint(data_)
        x = self.cg_op(fourier_op, csm, self.regularization_weight, zero_filled, x)
        return x
