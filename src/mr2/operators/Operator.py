"""General Operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, TypeAlias, cast

import torch
from typing_extensions import TypeVar, TypeVarTuple, Unpack, overload

import mr2.operators
from mr2.utils.TensorAttributeMixin import TensorAttributeMixin

Tin = TypeVarTuple('Tin')  # TODO: bind to torch.Tensors
Tin2 = TypeVarTuple('Tin2')  # TODO: bind to torch.Tensors
Tout = TypeVar('Tout', bound=tuple, covariant=True)  # TODO: bind to torch.Tensors


class Operator(Generic[Unpack[Tin], Tout], ABC, TensorAttributeMixin, torch.nn.Module):
    """The general Operator class.

    An operator is a function that maps one or more input tensors to one or more output tensors.
    Operators always return a tuple of tensors.
    Operators can be composed, added, multiplied, and applied to tensors.
    The forward method must be implemented by the subclasses.
    """

    @abstractmethod
    def forward(self, *args: Unpack[Tin]) -> Tout:
        """Apply forward operator."""
        ...

    def __call__(self, *args: Unpack[Tin]) -> Tout:
        """Apply the operator by calling the forward method."""
        return super().__call__(*args)

    def __matmul__(
        self: Operator[Unpack[Tin], Tout],
        other: Operator[Unpack[Tin2], tuple[Unpack[Tin]]] | Operator[Unpack[Tin2], tuple[torch.Tensor, ...]],
    ) -> Operator[Unpack[Tin2], Tout]:
        """Operator composition.

        Returns ``lambda x: self(other(x))``
        """
        return OperatorComposition(self, cast(Operator[Unpack[Tin2], tuple[Unpack[Tin]]], other))

    def __radd__(
        self: Operator[Unpack[Tin], tuple[Unpack[Tin]]], other: torch.Tensor | complex
    ) -> Operator[Unpack[Tin], tuple[Unpack[Tin]]]:
        """Operator right addition.

        Returns ``lambda x: other*x + self(x)``
        """
        return self + other

    @overload
    def __add__(self, other: Operator[Unpack[Tin], Tout]) -> Operator[Unpack[Tin], Tout]: ...
    @overload
    def __add__(
        self: Operator[Unpack[Tin], tuple[Unpack[Tin]]], other: torch.Tensor | complex
    ) -> Operator[Unpack[Tin], tuple[Unpack[Tin]]]: ...

    def __add__(
        self, other: Operator[Unpack[Tin], Tout] | torch.Tensor | complex | mr2.operators.ZeroOp
    ) -> Operator[Unpack[Tin], Tout] | Operator[Unpack[Tin], tuple[Unpack[Tin]]]:
        """Operator addition.

        Returns ``lambda x: self(x) + other(x)`` if other is a operator,
        ``lambda x: self(x) + other*x`` if other is a tensor
        """
        if isinstance(other, torch.Tensor | complex | int | float):
            s = cast(Operator[Unpack[Tin], tuple[Unpack[Tin]]], self)
            o = cast(Operator[Unpack[Tin], tuple[Unpack[Tin]]], mr2.operators.MultiIdentityOp() * other)
            return OperatorSum(s, o)
        elif isinstance(other, mr2.operators.ZeroOp):
            return self
        elif isinstance(other, Operator):
            return OperatorSum(
                cast(Operator[Unpack[Tin], Tout], other), self
            )  # cast due to https://github.com/python/mypy/issues/16335
        return NotImplemented

    def __mul__(self, other: torch.Tensor | complex) -> Operator[Unpack[Tin], Tout]:
        """Operator multiplication with tensor.

        Returns ``lambda x: self(x*other)``
        """
        return OperatorElementwiseProductLeft(self, other)

    def __rmul__(self, other: torch.Tensor | complex) -> Operator[Unpack[Tin], Tout]:
        """Operator multiplication with tensor.

        Returns ``lambda x: other*self(x)``
        """
        return OperatorElementwiseProductRight(self, other)

    @overload
    def __sub__(self, other: Operator[Unpack[Tin], Tout]) -> Operator[Unpack[Tin], Tout]: ...

    @overload
    def __sub__(
        self: Operator[Unpack[Tin], tuple[Unpack[Tin]]], other: torch.Tensor | complex
    ) -> Operator[Unpack[Tin], tuple[Unpack[Tin]]]: ...

    def __sub__(
        self, other: Operator[Unpack[Tin], Tout] | torch.Tensor | complex | mr2.operators.ZeroOp
    ) -> Operator[Unpack[Tin], Tout] | Operator[Unpack[Tin], tuple[Unpack[Tin]]]:
        """Operator subtraction.

        Returns ``lambda x: self(x) - other(x)`` if other is a operator,
        ``lambda x: self(x) - other*x`` if other is a tensor
        """
        if isinstance(other, mr2.operators.ZeroOp):
            return self
        return self + (-1.0) * other

    def __rsub__(
        self: Operator[Unpack[Tin], tuple[Unpack[Tin]]], other: torch.Tensor | complex
    ) -> Operator[Unpack[Tin], tuple[Unpack[Tin]]]:
        """Operator right subtraction.

        Returns ``lambda x: other*x - self(x)``
        """
        return (-1.0) * self + other


class OperatorComposition(Operator[Unpack[Tin2], Tout]):
    """Operator composition.

    Returns ``lambda x: operator1(operator2(x))``

    .. note::
        This is usually created by operator composition, e.g. ``operator1 @ operator2``.
    """

    def __init__(self, operator1: Operator[Unpack[Tin], Tout], operator2: Operator[Unpack[Tin2], tuple[Unpack[Tin]]]):
        """Operator composition initialization.

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

    def __call__(self, *args: Unpack[Tin2]) -> Tout:
        """Operator composition.

        Parameters
        ----------
        *args
            Input tensors for the inner operator.

        Returns
        -------
            Result of the composed operation.
        """
        return super().__call__(*args)

    def forward(self, *args: Unpack[Tin2]) -> Tout:
        """Apply forward of OperatorComposition.

        .. note::
            Prefer calling the instance of the OperatorComposition operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._operator1(*self._operator2(*args))


class OperatorSum(Operator[Unpack[Tin], Tout]):
    """Operator addition.

    Returns ``lambda x: operator1(x) + ... + operatorN(x)``

    .. note::
        This is usually created by operator addition, e.g. ``operator1 + operator2``.
    """

    _operators: list[Operator[Unpack[Tin], Tout]]  # actually a torch.nn.ModuleList

    def __init__(self, operator1: Operator[Unpack[Tin], Tout], /, *other_operators: Operator[Unpack[Tin], Tout]):
        """Operator addition initialization.

        Parameters
        ----------
        operator1
            First operator to add.
        *other_operators
            Additional operators to add.
        """
        super().__init__()
        ops: list[Operator[Unpack[Tin], Tout]] = []
        for op in (operator1, *other_operators):
            if isinstance(op, OperatorSum):
                ops.extend(op._operators)
            else:
                ops.append(op)
        self._operators = cast(list[Operator[Unpack[Tin], Tout]], torch.nn.ModuleList(ops))

    def __call__(self, *args: Unpack[Tin]) -> Tout:
        """Operator addition.

        Parameters
        ----------
        *args
            Input tensors to which the sum of operators is applied.

        Returns
        -------
            Result of the sum of operator applications.
        """
        return super().__call__(*args)

    def forward(self, *args: Unpack[Tin]) -> Tout:
        """Apply forward of OperatorSum.

        .. note::
            Prefer calling the instance of the OperatorSum operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """

        def _add(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> Tout:
            return cast(Tout, tuple(a_ + b_ for a_, b_ in zip(a, b, strict=True)))

        result = reduce(_add, (op(*args) for op in self._operators))
        return result


class OperatorElementwiseProductRight(Operator[Unpack[Tin], Tout]):
    """Operator elementwise right multiplication with a tensor.

    Performs ``Tensor*Operator(x)``

    .. note::
        This is usually created by operator multiplication with a tensor, e.g. ``tensor * operator``.
    """

    def __init__(self, operator: Operator[Unpack[Tin], Tout], scalar: torch.Tensor | complex):
        """Operator elementwise right multiplication initialization.

        Parameters
        ----------
        operator
            Operator to multiply with the scalar.
        scalar
            Scalar to multiply with the operator.
        """
        super().__init__()
        self._operator = operator
        self._scalar = scalar

    def __call__(self, *args: Unpack[Tin]) -> Tout:
        """Operator elementwise right multiplication.

        Parameters
        ----------
        *args
            Input tensors for the operator.

        Returns
        -------
            Result of the elementwise multiplication.
        """
        return super().__call__(*args)

    def forward(self, *args: Unpack[Tin]) -> Tout:
        """Apply forward of OperatorElementwiseProductRight.

        .. note::
            Prefer calling the instance of the OperatorElementwiseProductRight operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        out = self._operator(*args)
        return cast(Tout, tuple(a * self._scalar for a in out))


class OperatorElementwiseProductLeft(Operator[Unpack[Tin], Tout]):
    """Operator elementwise left multiplication  with a tensor.

    Performs ``Operator(x*Tensor)``

    .. note::
        This is usually created by operator multiplication with a tensor, e.g. `` operator * tensor``.
    """

    def __init__(self, operator: Operator[Unpack[Tin], Tout], scalar: torch.Tensor | complex):
        """Operator elementwise left multiplication initialization."""
        super().__init__()
        self._operator = operator
        self._scalar = scalar

    def __call__(self, *args: Unpack[Tin]) -> Tout:
        """Operator elementwise left multiplication.

        Parameters
        ----------
        *args
            Input tensors to be multiplied by the scalar before applying the operator.

        Returns
        -------
            Result of the operator application on the scaled tensors.
        """
        return super().__call__(*args)

    def forward(self, *args: Unpack[Tin]) -> Tout:
        """Apply forward of OperatorElementwiseProductLeft.

        .. note::
            Prefer calling the instance of the OperatorElementwiseProductLeft operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        multiplied = cast(tuple[Unpack[Tin]], tuple(a * self._scalar for a in args if isinstance(a, torch.Tensor)))
        out = self._operator(*multiplied)
        return cast(Tout, out)


OperatorType: TypeAlias = Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]
