"""Linear Operators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import overload

import torch

from mrpro.operators.Operator import (
    Operator,
    OperatorComposition,
    OperatorElementwiseProductLeft,
    OperatorElementwiseProductRight,
    OperatorSum,
    Tin2,
)


class LinearOperator(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """General Linear Operator.

    LinearOperators have exactly one input and one output,
    and fulfill f(a*x + b*y) = a*f(x) + b*f(y)
    with a,b scalars and x,y tensors.
    """

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
        callback: Callable | None = None,
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
        op_norm_old = torch.zeros(*tuple([1 for _ in range(vector.ndim)]))

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
    def __matmul__(self, other: Operator[*Tin2, tuple[torch.Tensor,]]) -> Operator[*Tin2, tuple[torch.Tensor,]]: ...

    def __matmul__(
        self, other: Operator[*Tin2, tuple[torch.Tensor,]] | LinearOperator
    ) -> Operator[*Tin2, tuple[torch.Tensor,]] | LinearOperator:
        """Operator composition.

        Returns lambda x: self(other(x))
        """
        if isinstance(other, LinearOperator):
            # LinearOperator@LinearOperator is linear
            return LinearOperatorComposition(self, other)
        else:
            return OperatorComposition(self, other)

    @overload
    def __add__(self, other: LinearOperator) -> LinearOperator: ...

    @overload
    def __add__(
        self, other: Operator[torch.Tensor, tuple[torch.Tensor,]]
    ) -> Operator[torch.Tensor, tuple[torch.Tensor,]]: ...

    def __add__(
        self, other: Operator[torch.Tensor, tuple[torch.Tensor,]] | LinearOperator
    ) -> Operator[torch.Tensor, tuple[torch.Tensor,]] | LinearOperator:
        """Operator addition.

        Returns lambda x: self(x) + other(x)
        """
        if not isinstance(other, LinearOperator):
            # general case
            return OperatorSum(self, other)  # other + cast(Operator[torch.Tensor, tuple[torch.Tensor,]], self)
        # Sum of linear operators is a linear operator
        return LinearOperatorSum(self, other)

    def __mul__(self, other: torch.Tensor) -> LinearOperator:
        """Operator elementwise left multiplication with tensor.

        Returns lambda x: self(other*x)
        """
        return LinearOperatorElementwiseProductLeft(self, other)

    def __rmul__(self, other: torch.Tensor) -> LinearOperator:
        """Operator elementwise right multiplication with tensor.

        Returns lambda x: other*self(x)
        """
        return LinearOperatorElementwiseProductRight(self, other)


class LinearOperatorComposition(LinearOperator, OperatorComposition[torch.Tensor, tuple[torch.Tensor,]]):
    """LinearOperator composition.

    Performs operator1(operator2(x))
    """

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator composition."""
        # (AB)^T = B^T A^T
        return self._operator2.adjoint(*self._operator1.adjoint(x))


class LinearOperatorSum(LinearOperator, OperatorSum[torch.Tensor, tuple[torch.Tensor,]]):
    """Operator addition."""

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator addition."""
        # (A+B)^T = A^T + B^T
        return (self._operator1.adjoint(x)[0] + self._operator2.adjoint(x)[0],)


class LinearOperatorElementwiseProductRight(
    LinearOperator, OperatorElementwiseProductRight[torch.Tensor, tuple[torch.Tensor,]]
):
    """Operator elementwise right multiplication with a tensor.

    Performs Tensor*LinearOperator(x)
    """

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor."""
        return self._operator.adjoint(x * self._tensor.conj())


class LinearOperatorElementwiseProductLeft(
    LinearOperator, OperatorElementwiseProductLeft[torch.Tensor, tuple[torch.Tensor,]]
):
    """Operator elementwise left multiplication with a tensor.

    Performs LinearOperator(Tensor*x)
    """

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor."""
        return (self._operator.adjoint(x)[0] * self._tensor.conj(),)


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
        return self.operator
