"""Endomorph Operators."""

import abc
import collections.abc
import typing

import torch

from mrpro.operators.Operator import Operator

Tin = typing.TypeVarTuple('Tin')
Tout = typing.TypeVar('Tout', bound=tuple[torch.Tensor, ...], covariant=True)
P = typing.ParamSpec('P')
Wrapped: typing.TypeAlias = collections.abc.Callable[P, Tout]
F = typing.TypeVar('F', bound=Wrapped)


class _EndomorphCallable(typing.Protocol):
    """A callable with the same number of tensor inputs and outputs.

    This is a protocol for a callable that takes a variadic number of tensor inputs
    and returns the same number of tensor outputs.

    This is only implemented for up to 10 inputs, if more inputs are given, the return
    will be a variadic number of tensors.

    This Protocol is used to decorate functions with the `endomorph` decorator.
    """

    @typing.overload
    def __call__(self, /) -> tuple[()]: ...
    @typing.overload
    def __call__(self, x1: torch.Tensor, /) -> tuple[torch.Tensor]: ...

    @typing.overload
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]: ...

    @typing.overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @typing.overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @typing.overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @typing.overload
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

    @typing.overload
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

    @typing.overload
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

    @typing.overload
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

    @typing.overload
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

    @typing.overload
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

    @typing.overload
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

    @abc.abstractmethod
    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the EndomorphOperator."""

    def __matmul__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]:
        """Operator composition."""
        return typing.cast(Operator[*Tin, Tout], super().__matmul__(other))

    def __rmatmul__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]:
        """Operator composition."""
        return other.__matmul__(typing.cast(Operator[*Tin, tuple[*Tin]], self))
