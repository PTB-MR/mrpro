"""General Operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, TypeVar, TypeVarTuple, cast, overload

import torch

import mrpro.operators

Tin = TypeVarTuple('Tin')  # TODO: bind to torch.Tensors
Tin2 = TypeVarTuple('Tin2')  # TODO: bind to torch.Tensors
Tout = TypeVar('Tout', bound=tuple, covariant=True)  # TODO: bind to torch.Tensors


class Operator(Generic[*Tin, Tout], ABC, torch.nn.Module):
    """The general Operator class."""

    @abstractmethod
    def forward(self, *args: *Tin) -> Tout:
        """Apply forward operator."""
        ...

    def __call__(self, *args: *Tin) -> Tout:
        """Operator call."""
        return super().__call__(*args)

    def __matmul__(self: Operator[*Tin, Tout], other: Operator[*Tin2, tuple[*Tin]]) -> Operator[*Tin2, Tout]:
        """Operator composition.

        Returns lambda x: self(other(x))
        """
        return OperatorComposition(self, other)

    def __radd__(self: Operator[*Tin, tuple[*Tin]], other: torch.Tensor) -> Operator[*Tin, tuple[*Tin]]:
        """Operator right addition.

        Returns lambda x: other*x + self(x)
        """
        return self + other

    @overload
    def __add__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]: ...
    @overload
    def __add__(self: Operator[*Tin, tuple[*Tin]], other: torch.Tensor) -> Operator[*Tin, tuple[*Tin]]: ...

    def __add__(
        self, other: Operator[*Tin, Tout] | torch.Tensor | mrpro.operators.ZeroOp
    ) -> Operator[*Tin, Tout] | Operator[*Tin, tuple[*Tin]]:
        """Operator addition.

        Returns lambda x: self(x) + other(x) if other is a operator,
        lambda x: self(x) + other*x if other is a tensor
        """
        if isinstance(other, torch.Tensor):
            s = cast(Operator[*Tin, tuple[*Tin]], self)
            o = cast(Operator[*Tin, tuple[*Tin]], mrpro.operators.MultiIdentityOp() * other)
            return OperatorSum(s, o)
        elif isinstance(other, mrpro.operators.ZeroOp):
            return self
        elif isinstance(other, Operator):
            return OperatorSum(
                cast(Operator[*Tin, Tout], other), self
            )  # cast due to https://github.com/python/mypy/issues/16335
        return NotImplemented  # type: ignore[unreachable]

    def __mul__(self, other: torch.Tensor | complex) -> Operator[*Tin, Tout]:
        """Operator multiplication with tensor.

        Returns lambda x: self(other*x)
        """
        return OperatorElementwiseProductLeft(self, other)

    def __rmul__(self, other: torch.Tensor | complex) -> Operator[*Tin, Tout]:
        """Operator multiplication with tensor.

        Returns lambda x: other*self(x)
        """
        return OperatorElementwiseProductRight(self, other)


class OperatorComposition(Operator[*Tin2, Tout]):
    """Operator composition."""

    def __init__(self, operator1: Operator[*Tin, Tout], operator2: Operator[*Tin2, tuple[*Tin]]):
        """Operator composition initialization.

        Returns lambda x: operator1(operator2(x))

        Parameters
        ----------
        operator1
            outer operator
        operator2
            inner operator
        """
        super().__init__()
        self._operator1 = operator1
        self._operator2 = operator2

    def forward(self, *args: *Tin2) -> Tout:
        """Operator composition."""
        return self._operator1(*self._operator2(*args))


class OperatorSum(Operator[*Tin, Tout]):
    """Operator addition."""

    _operators: list[Operator[*Tin, Tout]]  # actually a torch.nn.ModuleList

    def __init__(self, operator1: Operator[*Tin, Tout], /, *other_operators: Operator[*Tin, Tout]):
        """Operator addition initialization."""
        super().__init__()
        ops: list[Operator[*Tin, Tout]] = []
        for op in (operator1, *other_operators):
            if isinstance(op, OperatorSum):
                ops.extend(op._operators)
            else:
                ops.append(op)
        self._operators = cast(list[Operator[*Tin, Tout]], torch.nn.ModuleList(ops))

    def forward(self, *args: *Tin) -> Tout:
        """Operator addition."""

        def _add(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> Tout:
            return cast(Tout, tuple(a_ + b_ for a_, b_ in zip(a, b, strict=True)))

        result = reduce(_add, (op(*args) for op in self._operators))
        return result


class OperatorElementwiseProductRight(Operator[*Tin, Tout]):
    """Operator elementwise right multiplication with a tensor.

    Performs Tensor*Operator(x)
    """

    def __init__(self, operator: Operator[*Tin, Tout], scalar: torch.Tensor | complex):
        """Operator elementwise right multiplication initialization."""
        super().__init__()
        self._operator = operator
        self._scalar = scalar

    def forward(self, *args: *Tin) -> Tout:
        """Operator elementwise right multiplication."""
        out = self._operator(*args)
        return cast(Tout, tuple(a * self._scalar for a in out))


class OperatorElementwiseProductLeft(Operator[*Tin, Tout]):
    """Operator elementwise left multiplication  with a tensor.

    Performs Operator(x*Tensor)
    """

    def __init__(self, operator: Operator[*Tin, Tout], scalar: torch.Tensor | complex):
        """Operator elementwise left multiplication initialization."""
        super().__init__()
        self._operator = operator
        self._scalar = scalar

    def forward(self, *args: *Tin) -> Tout:
        """Operator elementwise left multiplication."""
        multiplied = cast(tuple[*Tin], tuple(a * self._scalar for a in args if isinstance(a, torch.Tensor)))
        out = self._operator(*multiplied)
        return cast(Tout, out)
