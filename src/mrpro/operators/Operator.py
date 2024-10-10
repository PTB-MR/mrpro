"""General Operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TypeVarTuple, cast

import torch

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

    def __add__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]:
        """Operator addition.

        Returns lambda x: self(x) + other(x)
        """
        return OperatorSum(self, other)

    def __mul__(self, other: torch.Tensor) -> Operator[*Tin, Tout]:
        """Operator multiplication with tensor.

        Returns lambda x: self(other*x)
        """
        return OperatorElementwiseProductLeft(self, other)

    def __rmul__(self, other: torch.Tensor) -> Operator[*Tin, Tout]:
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

    def __init__(self, operator1: Operator[*Tin, Tout], operator2: Operator[*Tin, Tout]):
        """Operator addition initialization."""
        super().__init__()
        self._operator1 = operator1
        self._operator2 = operator2

    def forward(self, *args: *Tin) -> Tout:
        """Operator addition."""
        return cast(Tout, tuple(a + b for a, b in zip(self._operator1(*args), self._operator2(*args), strict=True)))


class OperatorElementwiseProductRight(Operator[*Tin, Tout]):
    """Operator elementwise right multiplication with a tensor.

    Performs Tensor*Operator(x)
    """

    def __init__(self, operator: Operator[*Tin, Tout], tensor: torch.Tensor):
        """Operator elementwise right multiplication initialization."""
        super().__init__()
        self._operator = operator
        self._tensor = tensor

    def forward(self, *args: *Tin) -> Tout:
        """Operator elementwise right multiplication."""
        out = self._operator(*args)
        return cast(Tout, tuple(a * self._tensor for a in out))


class OperatorElementwiseProductLeft(Operator[*Tin, Tout]):
    """Operator elementwise left multiplication  with a tensor.

    Performs Operator(x*Tensor)
    """

    def __init__(self, operator: Operator[*Tin, Tout], tensor: torch.Tensor):
        """Operator elementwise left multiplication initialization."""
        super().__init__()
        self._operator = operator
        self._tensor = tensor

    def forward(self, *args: *Tin) -> Tout:
        """Operator elementwise left multiplication."""
        multiplied = cast(tuple[*Tin], tuple(a * self._tensor for a in args))
        out = self._operator(*multiplied)
        return cast(Tout, out)
