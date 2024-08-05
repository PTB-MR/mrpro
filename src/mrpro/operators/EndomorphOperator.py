"""Endomorph Operators."""

from abc import abstractmethod
from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeAlias, TypeVar, TypeVarTuple, cast, overload

import torch

from mrpro.operators.Operator import Operator

Tin = TypeVarTuple('Tin')
Tout = TypeVar('Tout', bound=tuple[torch.Tensor, ...], covariant=True)
P = ParamSpec('P')
Wrapped: TypeAlias = Callable[P, Tout]
F = TypeVar('F', bound=Wrapped)


class _EndomorphCallable(Protocol):
    """A callable with the same number of tensor inputs and outputs.

    This is a protocol for a callable that takes a variadic number of tensor inputs
    and returns the same number of tensor outputs.

    This is only implemented for up to 10 inputs, if more inputs are given, the return
    will be a variadic number of tensors.

    This Protocol is used to decorate functions with the `endomorph` decorator.
    """

    @overload
    def __call__(self, /) -> tuple[()]: ...
    @overload
    def __call__(self, x1: torch.Tensor, /) -> tuple[torch.Tensor]: ...

    @overload
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        /,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        x9: torch.Tensor,
        /,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        x9: torch.Tensor,
        x10: torch.Tensor,
        /,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        x9: torch.Tensor,
        x10: torch.Tensor,
        /,
        *args: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        *tuple[torch.Tensor, ...],
    ]: ...

    @overload
    def __call__(self, /, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

    def __call__(self, /, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


def endomorph(f: F, /) -> _EndomorphCallable:
    """Decorate a function to make it an endomorph callable.

    This adds overloads for N->N-Tensor signatures, for N<10.
    For >10 inputs, the return type will a tuple with >10 tensors.
    """
    return f


class EndomorphOperator(Operator[*tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]):
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

    def __matmul__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]:
        """Operator composition."""
        return cast(Operator[*Tin, Tout], super().__matmul__(other))

    def __rmatmul__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]:
        """Operator composition."""
        return other.__matmul__(cast(Operator[*Tin, tuple[*Tin]], self))
