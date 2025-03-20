"""Endomorph Operators."""

from __future__ import annotations

from abc import abstractmethod
from typing import cast

import torch
from typing_extensions import Any, TypeVar, TypeVarTuple, Unpack, overload

import mrpro.operators
from mrpro.operators.Operator import Operator
from mrpro.utils.typing import endomorph

Tin = TypeVarTuple('Tin')
Tout = TypeVar('Tout', bound=tuple[torch.Tensor, ...], covariant=True)


class EndomorphOperator(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]):
    """Endomorph Operator.

    Endomorph Operators have N tensor inputs and exactly N outputs.
    """

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the EndomorphOperator."""
        # This function only overwrites the type hints of the base operator class
        return super().__call__(*x)

    @abstractmethod
    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the EndomorphOperator."""

    @overload
    def __matmul__(self, other: EndomorphOperator) -> EndomorphOperator: ...
    @overload
    def __matmul__(self, other: Operator[Unpack[Tin], Tout]) -> Operator[Unpack[Tin], Tout]: ...

    def __matmul__(
        self, other: Operator[Unpack[Tin], Tout] | EndomorphOperator
    ) -> Operator[Unpack[Tin], Tout] | EndomorphOperator:
        """Operator composition."""
        if isinstance(other, mrpro.operators.MultiIdentityOp):
            return self
        elif isinstance(self, mrpro.operators.MultiIdentityOp):
            return other

        res = super().__matmul__(cast(Any, other))  # avoid mypy 1.11 crash
        if isinstance(other, EndomorphOperator):
            return cast(EndomorphOperator, res)
        else:
            return cast(Operator[Unpack[Tin], Tout], res)

    def __rmatmul__(self, other: Operator[Unpack[Tin], Tout]) -> Operator[Unpack[Tin], Tout]:
        """Operator composition."""
        return other.__matmul__(cast(Operator[Unpack[Tin], tuple[Unpack[Tin]]], self))
