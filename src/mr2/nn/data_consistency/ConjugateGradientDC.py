"""Conjugate gradient data consistency."""

from typing import overload

import torch
from torch.nn import Module, Parameter

from mr2.data.CsmData import CsmData
from mr2.data.KData import KData
from mr2.operators.ConjugateGradientOp import ConjugateGradientOp
from mr2.operators.FourierOp import FourierOp
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.SensitivityOp import SensitivityOp


class ConjugateGradientDC(Module):
    """Conjugate gradient data consistency."""

    def __init__(self, initial_regularization_weight: torch.Tensor | float):
        """Initialize the conjugate gradient data consistency.

        Parameters
        ----------
        initial_regularization_weight
            Initial regularization weight. The regularization weight is a trainable parameter.
            Must be a positive scalar.
        """
        super().__init__()
        weight = torch.as_tensor(initial_regularization_weight)
        if weight.ndim != 0:
            raise ValueError('Regularization weight must be a scalar')
        if weight.item() <= 0:
            raise ValueError('Regularization weight must be positive')
        self.log_weight = Parameter(weight.log())

    @overload
    def __call__(
        self,
        image: torch.Tensor,
        data: KData,
        fourier_op: FourierOp | None = None,
        csm: torch.Tensor | CsmData | None = None,
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self,
        image: torch.Tensor,
        data: torch.Tensor,
        fourier_op: FourierOp,
        csm: torch.Tensor | CsmData | None = None,
    ) -> torch.Tensor: ...

    def __call__(
        self,
        image: torch.Tensor,
        data: KData | torch.Tensor,
        fourier_op: LinearOperator | None = None,
        csm: torch.Tensor | CsmData | None = None,
    ) -> torch.Tensor:
        """Apply the data consistency.

        Parameters
        ----------
        image
            Current image estimate.
        data
            k-space data.
        fourier_op
            Fourier operator matching the k-space data. If None and data is provided as a `~mr2.data.KData` object,
            the Fourier operator is automatically created from the data.
            This operator can already include the coil sensitivity weighting, if gradients wrt the coil sensitivity maps
            NOT required. Otherwise, they should be given as an additional argument.
        csm
            Coil sensitivity maps. If None, no coil sensitivity weighting is applied.

        Returns
        -------
            Updated image estimate.
        """
        return super().__call__(image, data, fourier_op, csm)

    def forward(
        self,
        image: torch.Tensor,
        data: torch.Tensor | KData,
        fourier_op: FourierOp | None = None,
        csm: torch.Tensor | CsmData | None = None,
    ) -> torch.Tensor:
        """Apply the data consistency."""
        if fourier_op is None:
            if isinstance(data, KData):
                fourier_op = FourierOp.from_kdata(data)
            else:
                raise ValueError('Either a KData or a FourierOp is required')

        data_ = data.data if isinstance(data, KData) else data

        if csm is None:
            csm = torch.tensor(())
        elif isinstance(csm, CsmData):
            csm = csm.data

        def operator_factory(csm: torch.Tensor, weight: torch.Tensor, *_):
            op = fourier_op.gram
            if csm.numel():
                csm_op = SensitivityOp(csm)
                op = csm_op.H @ op @ csm_op
            op = op + weight
            return op

        def rhs_factory(_csm: torch.Tensor, weight: torch.Tensor, zero_filled: torch.Tensor, image: torch.Tensor):
            return (zero_filled + weight * image,)

        cg_op = ConjugateGradientOp(operator_factory=operator_factory, rhs_factory=rhs_factory)
        (result,) = cg_op(csm, self.log_weight.exp(), fourier_op.adjoint(data_)[0], image)
        return result
