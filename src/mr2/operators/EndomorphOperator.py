"""Endomorph Operators."""

from __future__ import annotations

from abc import abstractmethod
from typing import cast

import torch
from typing_extensions import Any, TypeVar, TypeVarTuple, Unpack, overload

import mr2.operators
from mr2.operators.Operator import Operator
from mr2.utils.typing import endomorph

Tin = TypeVarTuple('Tin')
Tout = TypeVar('Tout', bound=tuple[torch.Tensor, ...], covariant=True)


class EndomorphOperator(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]):
    """Endomorph Operator.

    Endomorph Operators have N tensor inputs and exactly N outputs.
    """

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the endomorphism.

        An endomorphism is an operator that maps a set of tensors
        to another set of tensors of the same number.

        Parameters
        ----------
        *x
            One or more input tensors.

        Returns
        -------
            A tuple containing the same number of tensors as input,
            resulting from the operator's action.
        """
        # This function only overwrites the type hints of the base operator class
        return super().__call__(*x)

    @abstractmethod
    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply forward of EndomorphOperator.

        .. note::
            Prefer calling the instance of the EndomorphOperator operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """

    @overload
    def __matmul__(self, other: EndomorphOperator) -> EndomorphOperator: ...
    @overload
    def __matmul__(self, other: Operator[Unpack[Tin], Tout]) -> Operator[Unpack[Tin], Tout]: ...

    def __matmul__(
        self, other: Operator[Unpack[Tin], Tout] | EndomorphOperator
    ) -> Operator[Unpack[Tin], Tout] | EndomorphOperator:
        """Operator composition."""
        if isinstance(other, mr2.operators.MultiIdentityOp):
            return self
        elif isinstance(self, mr2.operators.MultiIdentityOp):
            return other

        res = super().__matmul__(cast(Any, other))  # avoid mypy 1.11 crash
        if isinstance(other, EndomorphOperator):
            return cast(EndomorphOperator, res)
        else:
            return cast(Operator[Unpack[Tin], Tout], res)

    def __rmatmul__(self, other: Operator[Unpack[Tin], Tout]) -> Operator[Unpack[Tin], Tout]:
        """Operator composition."""
        return other.__matmul__(cast(Operator[Unpack[Tin], tuple[Unpack[Tin]]], self))
