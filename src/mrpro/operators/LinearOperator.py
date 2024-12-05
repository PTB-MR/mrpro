"""Linear Operators."""

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import reduce
from typing import cast, no_type_check

import torch
from typing_extensions import Any, Unpack, overload

import mrpro.operators
from mrpro.operators.Operator import (
    Operator,
    OperatorComposition,
    OperatorElementwiseProductLeft,
    OperatorElementwiseProductRight,
    OperatorSum,
    Tin2,
)


class _AutogradWrapper(torch.autograd.Function):
    """Wrap forward and adjoint functions for autograd."""

    # If the forward and adjoint implementation are vmap-compatible,
    # the function can be marked as such to enable vmap support.
    generate_vmap_rule = True

    @staticmethod
    def forward(
        fw: Callable[[torch.Tensor], torch.Tensor],
        bw: Callable[[torch.Tensor], torch.Tensor],  # noqa: ARG004
        x: torch.Tensor,
    ) -> torch.Tensor:
        return fw(x)

    @staticmethod
    def setup_context(
        ctx: Any,  # noqa: ANN401
        inputs: tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        ctx.fw, ctx.bw, x = inputs
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> tuple[None, None, torch.Tensor]:  # noqa: ANN401
        return None, None, _AutogradWrapper.apply(ctx.bw, ctx.fw, grad_output[0])

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> torch.Tensor:  # noqa: ANN401
        return _AutogradWrapper.apply(ctx.fw, ctx.bw, grad_inputs[-1])


class LinearOperator(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """General Linear Operator.

    LinearOperators have exactly one input and one output,
    and fulfill f(a*x + b*y) = a*f(x) + b*f(y)
    with a,b scalars and x,y tensors.
    """

    @no_type_check
    def __init_subclass__(cls, adjoint_as_backward: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        """Wrap the forward and adjoint functions for autograd.

        Parameters
        ----------
        adjoint_as_backward
            if True, the adjoint function is used as the backward function,
            else automatic differentiation of the forward function is used.
        kwargs
            additional keyword arguments, passed to the super class
        """
        if adjoint_as_backward and not hasattr(cls, '_saved_forward'):
            cls._saved_forward, cls._saved_adjoint = cls.forward, cls.adjoint
            cls.forward = lambda self, x: (
                _AutogradWrapper.apply(lambda x: self._saved_forward(x)[0], lambda x: self._saved_adjoint(x)[0], x),
            )
            cls.adjoint = lambda self, x: (
                _AutogradWrapper.apply(lambda x: self._saved_adjoint(x)[0], lambda x: self._saved_forward(x)[0], x),
            )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator."""
        ...

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        """Adjoint operator."""
        return AdjointLinearOperator(self)

    def operator_norm(
        self,
        initial_value: torch.Tensor,
        dim: Sequence[int] | None,
        max_iterations: int = 20,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-5,
        callback: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Power iteration for computing the operator norm of the linear operator.

        Parameters
        ----------
        initial_value
            initial value to start the iteration; if the initial value contains a zero-vector for
            one of the considered problems, the function throws an value error.
        dim
            the dimensions of the tensors on which the operator operates.
            For example, for a matrix-vector multiplication example, a batched matrix tensor with shape (4,30,80,160),
            input tensors of shape (4,30,160) to be multiplied, and dim = None, it is understood that the
            matrix representation of the operator corresponds to a block diagonal operator (with 4*30 matrices)
            and thus the algorithm returns a tensor of shape (1,1,1) containing one single value.
            In contrast, if for example, dim=(-1,), the algorithm computes a batched operator
            norm and returns a tensor of shape (4,30,1) corresponding to the operator norms of the respective
            matrices in the diagonal of the block-diagonal operator (if considered in matrix representation).
            In any case, the output of the algorithm has the same number of dimensions as the elements of the
            domain of the considered operator (whose dimensionality is implicitly defined by choosing dim), such that
            the pointwise multiplication of the operator norm and elements of the domain (to be for example used
            in a Landweber iteration) is well-defined.
        max_iterations
            maximum number of iterations
        relative_tolerance
            absolute tolerance for the change of the operator-norm at each iteration;
            if set to zero, the maximal number of iterations is the only stopping criterion used to stop
            the power iteration
        absolute_tolerance
            absolute tolerance for the change of the operator-norm at each iteration;
            if set to zero, the maximal number of iterations is the only stopping criterion used to stop
            the power iteration
        callback
            user-provided function to be called at each iteration

        Returns
        -------
            an estimaton of the operator norm
        """
        if max_iterations < 1:
            raise ValueError('The number of iterations should be larger than zero.')

        # check that the norm of the starting value is not zero
        norm_initial_value = torch.linalg.vector_norm(initial_value, dim=dim, keepdim=True)
        if not (norm_initial_value > 0).all():
            if dim is None:
                raise ValueError('The initial value for the iteration should be different from the zero-vector.')
            else:
                raise ValueError(
                    'Found at least one zero-vector as starting point. \
                    For each of the considered operators, the initial value for the iteration \
                    should be different from the zero-vector.'
                )

        # set initial value
        vector = initial_value

        # creaty dummy operator norm value that cannot be correct because by definition, the
        # operator norm is a strictly positive number. This ensures that the first time the
        # change between the old and the new estimate of the operator norm is non-zero and
        # thus prevents the loop from exiting despite a non-correct estimate.
        op_norm_old = torch.zeros(*tuple([1 for _ in range(vector.ndim)]), device=vector.device)

        dim = tuple(dim) if dim is not None else dim
        for _ in range(max_iterations):
            # apply the operator to the vector
            (vector_new,) = self.adjoint(*self(vector))

            # compute estimate of the operator norm
            product = vector.real * vector_new.real
            if vector.is_complex() and vector_new.is_complex():
                product += vector.imag * vector_new.imag
            op_norm = product.sum(dim, keepdim=True).sqrt()

            # check if stopping criterion is fulfillfed; if not continue the iteration
            if (absolute_tolerance > 0 or relative_tolerance > 0) and torch.isclose(
                op_norm, op_norm_old, atol=absolute_tolerance, rtol=relative_tolerance
            ).all():
                break

            # normalize vector
            vector = vector_new / torch.linalg.vector_norm(vector_new, dim=dim, keepdim=True)
            op_norm_old = op_norm

            if callback is not None:
                callback(op_norm)

        return op_norm

    @overload
    def __matmul__(self, other: LinearOperator) -> LinearOperator: ...

    @overload
    def __matmul__(
        self, other: Operator[Unpack[Tin2], tuple[torch.Tensor,]]
    ) -> Operator[Unpack[Tin2], tuple[torch.Tensor,]]: ...

    def __matmul__(
        self, other: Operator[Unpack[Tin2], tuple[torch.Tensor,]] | LinearOperator
    ) -> Operator[Unpack[Tin2], tuple[torch.Tensor,]] | LinearOperator:
        """Operator composition.

        Returns lambda x: self(other(x))
        """
        if isinstance(other, mrpro.operators.IdentityOp):
            # neutral element of composition
            return self
        elif isinstance(self, mrpro.operators.IdentityOp):
            return other
        elif isinstance(other, LinearOperator):
            # LinearOperator@LinearOperator is linear
            return LinearOperatorComposition(self, other)
        elif isinstance(other, Operator):
            # cast due to https://github.com/python/mypy/issues/16335
            return OperatorComposition(self, cast(Operator[Unpack[Tin2], tuple[torch.Tensor,]], other))
        return NotImplemented  # type: ignore[unreachable]

    def __radd__(self, other: torch.Tensor) -> LinearOperator:
        """Operator addition.

        Returns lambda self(x) + other*x
        """
        return self + other

    @overload  # type: ignore[override]
    def __add__(self, other: LinearOperator | torch.Tensor) -> LinearOperator: ...

    @overload
    def __add__(
        self, other: Operator[torch.Tensor, tuple[torch.Tensor,]]
    ) -> Operator[torch.Tensor, tuple[torch.Tensor,]]: ...

    def __add__(
        self, other: Operator[torch.Tensor, tuple[torch.Tensor,]] | LinearOperator | torch.Tensor
    ) -> Operator[torch.Tensor, tuple[torch.Tensor,]] | LinearOperator:
        """Operator addition.

        Returns lambda x: self(x) + other(x) if other is a operator,
        lambda x: self(x) + other if other is a tensor
        """
        if isinstance(other, torch.Tensor):
            # tensor addition
            return LinearOperatorSum(self, mrpro.operators.IdentityOp() * other)
        elif isinstance(self, mrpro.operators.ZeroOp):
            # neutral element of addition
            return other
        elif isinstance(other, mrpro.operators.ZeroOp):
            # neutral element of addition
            return self
        elif isinstance(other, LinearOperator):
            # sum of LinearOperators is linear
            return LinearOperatorSum(self, other)
        elif isinstance(other, Operator):
            # for general operators
            return OperatorSum(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __mul__(self, other: torch.Tensor | complex) -> LinearOperator:
        """Operator elementwise left multiplication with tensor/scalar.

        Returns lambda x: self(x*other)
        """
        if isinstance(other, complex | float | int):
            if other == 0:
                return mrpro.operators.ZeroOp()
            if other == 1:
                return self
            else:
                return LinearOperatorElementwiseProductLeft(self, other)
        elif isinstance(other, torch.Tensor):
            return LinearOperatorElementwiseProductLeft(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __rmul__(self, other: torch.Tensor | complex) -> LinearOperator:
        """Operator elementwise right multiplication with tensor/scalar.

        Returns lambda x: other*self(x)
        """
        if isinstance(other, complex | float | int):
            if other == 0:
                return mrpro.operators.ZeroOp()
            if other == 1:
                return self
            else:
                return LinearOperatorElementwiseProductRight(self, other)
        elif isinstance(other, torch.Tensor):
            return LinearOperatorElementwiseProductRight(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __and__(self, other: LinearOperator) -> mrpro.operators.LinearOperatorMatrix:
        """Vertical stacking of two LinearOperators."""
        if not isinstance(other, LinearOperator):
            return NotImplemented  # type: ignore[unreachable]
        operators = [[self], [other]]
        return mrpro.operators.LinearOperatorMatrix(operators)

    def __or__(self, other: LinearOperator) -> mrpro.operators.LinearOperatorMatrix:
        """Horizontal stacking of two LinearOperators."""
        if not isinstance(other, LinearOperator):
            return NotImplemented  # type: ignore[unreachable]
        operators = [[self, other]]
        return mrpro.operators.LinearOperatorMatrix(operators)

    @property
    def gram(self) -> LinearOperator:
        """Gram operator.

        For a LinearOperator :math:`A`, the self-adjoint Gram operator is defined as :math:`A^H A`.

        Note: This is a default implementation that can be overwritten by subclasses for more efficient
        implementations.
        """
        return self.H @ self


class LinearOperatorComposition(
    LinearOperator,
    OperatorComposition[torch.Tensor, tuple[torch.Tensor,]],
):
    """LinearOperator composition.

    Performs operator1(operator2(x))
    """

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator composition."""
        # (AB)^H = B^H A^H
        return self._operator2.adjoint(*self._operator1.adjoint(x))

    @property
    def gram(self) -> LinearOperator:
        """Gram operator."""
        # (AB)^H(AB) = B^H (A^H A) B
        return self._operator2.H @ self._operator1.gram @ self._operator2


class LinearOperatorSum(LinearOperator, OperatorSum[torch.Tensor, tuple[torch.Tensor,]]):
    """Operator addition."""

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator addition."""
        # (A+B)^H = A^H + B^H
        return (reduce(operator.add, (op.adjoint(x)[0] for op in self._operators)),)


class LinearOperatorElementwiseProductRight(
    LinearOperator, OperatorElementwiseProductRight[torch.Tensor, tuple[torch.Tensor,]]
):
    """Operator elementwise right multiplication with a tensor.

    Performs Tensor*LinearOperator(x)
    """

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor/scalar."""
        conj = self._scalar.conj() if isinstance(self._scalar, torch.Tensor) else self._scalar.conjugate()
        return self._operator.adjoint(x * conj)

    @property
    def gram(self) -> LinearOperator:
        """Gram Operator."""
        if isinstance(self._scalar, torch.Tensor):
            factor: torch.Tensor | complex = self._scalar.conj() * self._scalar
            if self._scalar.numel() > 1:
                # only scalars can be moved outside the linear operator
                return self._operator.H @ (factor * self._operator)
        else:
            factor = self._scalar.conjugate() * self._scalar
        return factor * self._operator.gram


class LinearOperatorElementwiseProductLeft(
    LinearOperator, OperatorElementwiseProductLeft[torch.Tensor, tuple[torch.Tensor,]]
):
    """Operator elementwise left multiplication with a tensor.

    Performs LinearOperator(Tensor*x)
    """

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor/scalar."""
        conj = self._scalar.conj() if isinstance(self._scalar, torch.Tensor) else self._scalar.conjugate()
        return (self._operator.adjoint(x)[0] * conj,)

    @property
    def gram(self) -> LinearOperator:
        """Gram Operator."""
        conj = self._scalar.conj() if isinstance(self._scalar, torch.Tensor) else self._scalar.conjugate()
        return conj * self._operator.gram * self._scalar


class AdjointLinearOperator(LinearOperator):
    """Adjoint of a LinearOperator."""

    def __init__(self, operator: LinearOperator) -> None:
        """Initialize the adjoint of a LinearOperator."""
        super().__init__()
        self._operator = operator

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the original LinearOperator."""
        return self._operator.adjoint(x)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the adjoint, i.e. the original LinearOperator."""
        return self._operator.forward(x)

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        """Adjoint of adjoint operator, i.e. original LinearOperator."""
        return self._operator
