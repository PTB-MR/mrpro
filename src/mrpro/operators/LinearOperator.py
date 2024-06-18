"""Linear Operators."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import overload

import torch

from mrpro.operators.Operator import Operator
from mrpro.operators.Operator import OperatorComposition
from mrpro.operators.Operator import OperatorElementwiseProductLeft
from mrpro.operators.Operator import OperatorElementwiseProductRight
from mrpro.operators.Operator import OperatorSum
from mrpro.operators.Operator import Tin2


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
        max_iterations: int = 64,
    ) -> torch.Tensor:
        """Power iteration for computing the operator norm of the linear operator.

        Parameters
        ----------
        initial_value
            initial value to start the iteration; if chosen exactly as zero-tensor,
            we randomly generate an initial value.
        dim
            the dimensions of the tensors on which the operator operates.
            For example, for a matrix-vector multiplication example, a batched matrix tensor with shape (4,30,80,160),
            input tensors of shape (4,30,160) to be multiplied, and dim = None, it is understood that the the
            matrix representation of the operator corresponds to a block diagonal operator (with 4*30 matrices)
            and thus the algorithm returns a tensor of shape (1,1,1) containing one single value.
            In contrast, if for example, dim=(-1,), the algorithm computes a batched operator
            norm and returns a tensor of shape (4,30,1) corresponding to the operator norms of the respective
            matrices in the diagonal of the block-diagonal operator (if considered in matrix representation).
            In any case, the output of the algorithm has the same number of dimensions as the elements of the
            domain of the considered operator (whose dimensionality is implicitly defined by choosing dim), such that
            the pointwise multiplication of the operator norm and elements of the domain (to be for example used
            in a Landweber iteration) is well-defined.
        max_iterations, optional
            maximal number of iterations

        Returns
        -------
            an estimaton of the operator norm
        """
        # check if initial value is exactly zero; if yes, set it to a random value
        vector = torch.randn_like(initial_value) if (initial_value == 0.0).all() else initial_value

        for _ in range(max_iterations):
            # apply the operator to the vector
            (vector,) = self.adjoint(*self(vector))

            # normalize vector
            vector /= torch.linalg.vector_norm(vector, dim=dim, keepdim=True)

        (operator_vector,) = self.adjoint(*self(vector))

        product = vector.real * operator_vector.real
        if vector.is_complex() and operator_vector.is_complex():
            product += vector.imag * operator_vector.imag
        dim = tuple(dim) if dim is not None else dim
        op_norm = product.sum(dim, keepdim=True).sqrt()

        return op_norm

    @overload  # type: ignore[override]
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

    def __rmul__(self, other: torch.Tensor) -> LinearOperator:  # type: ignore[misc]
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

    Peforms Tensor*LinearOperator(x)
    """

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor."""
        return self._operator.adjoint(x * self._tensor.conj())


class LinearOperatorElementwiseProductLeft(
    LinearOperator, OperatorElementwiseProductLeft[torch.Tensor, tuple[torch.Tensor,]]
):
    """Operator elementwise left multiplication with a tensor.

    Peforms LinearOperator(Tensor*x)
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
