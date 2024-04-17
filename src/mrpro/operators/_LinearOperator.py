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
from typing import overload

import torch

from mrpro.operators._Operator import Operator
from mrpro.operators._Operator import OperatorComposition
from mrpro.operators._Operator import OperatorElementwiseProductLeft
from mrpro.operators._Operator import OperatorElementwiseProductRight
from mrpro.operators._Operator import OperatorSum
from mrpro.operators._Operator import Tin2


# LinearOperators have exactly one input and one output,
# and fulfill f(a*x + b*y) = a*f(x) + b*f(y)
# with a,b scalars and x,y tensors.
class LinearOperator(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """General Linear Operator."""

    @abstractmethod
    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator."""
        ...

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        """Adjoint operator."""
        return AdjointLinearOperator(self)

    @overload  # type: ignore[override]
    def __matmul__(self, other: LinearOperator) -> LinearOperator: ...

    @overload
    def __matmul__(self, other: Operator[*Tin2, tuple[torch.Tensor,]]) -> Operator[*Tin2, tuple[torch.Tensor,]]: ...

    def __matmul__(
        self, other: Operator[*Tin2, tuple[torch.Tensor,]] | LinearOperator
    ) -> Operator[*Tin2, tuple[torch.Tensor,]] | LinearOperator:
        """Operator composition."""
        if isinstance(other, LinearOperator):
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
        """Operator addition."""
        if not isinstance(other, LinearOperator):
            # general case
            return OperatorSum(self, other)  # other + cast(Operator[torch.Tensor, tuple[torch.Tensor,]], self)
        return LinearOperatorSum(self, other)

    def __mul__(self, other: torch.Tensor) -> LinearOperator:
        """Operator elementwise left multiplication with tensor."""
        return LinearOperatorElementwiseProductLeft(self, other)

    def __rmul__(self, other: torch.Tensor) -> LinearOperator:  # type: ignore[misc]
        """Operator elementwise right multiplication with tensor."""
        return LinearOperatorElementwiseProductRight(self, other)


class LinearOperatorComposition(LinearOperator, OperatorComposition[torch.Tensor, tuple[torch.Tensor,]]):
    """Operator composition."""

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator composition."""
        return self._operator2.adjoint(*self._operator1.adjoint(x))


class LinearOperatorSum(LinearOperator, OperatorSum[torch.Tensor, tuple[torch.Tensor,]]):
    """Operator addition."""

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator addition."""
        return (self._operator1.adjoint(x)[0] + self._operator2.adjoint(x)[0],)


class LinearOperatorElementwiseProductRight(
    LinearOperator, OperatorElementwiseProductRight[torch.Tensor, tuple[torch.Tensor,]]
):
    """Operator elementwise right multiplication with a tensor."""

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor."""
        if self._tensor.is_complex():
            return self._operator.adjoint(x * self._tensor.conj())
        return self._operator.adjoint(x * self._tensor)


class LinearOperatorElementwiseProductLeft(
    LinearOperator, OperatorElementwiseProductLeft[torch.Tensor, tuple[torch.Tensor,]]
):
    """Operator elementwise left multiplication with a tensor."""

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor."""
        if self._tensor.is_complex():
            return (self._operator.adjoint(x)[0] * self._tensor.conj(),)
        return (self._operator.adjoint(x)[0] * self._tensor,)


class AdjointLinearOperator(LinearOperator):
    """Adjoint of a LinearOperator."""

    def __init__(self, operator: LinearOperator) -> None:
        super().__init__()
        self._operator = operator

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator."""
        return self._operator.adjoint(x)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Operator."""
        return self._operator.forward(x)

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        """Adjoint of adjoint operator."""
        return self.operator
