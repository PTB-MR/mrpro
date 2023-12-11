"""Linear Operators."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from abc import abstractmethod

import torch

from mrpro.operators import Operator
from mrpro.operators._Operator import OperatorComposition
from mrpro.operators._Operator import OperatorElementwiseProduct
from mrpro.operators._Operator import OperatorSum
from mrpro.operators._Operator import Ts


class LinearOperator(Operator[*Ts]):
    """General Linear Operator."""

    @abstractmethod
    def adjoint(self, *args: *Ts): ...

    @property
    def H(self):
        """Adjoint operator."""
        return AdjointLinearOperator(self)

    def __matmul__(self, other: Operator):
        """Operator composition."""
        if not isinstance(other, LinearOperator):
            return Operator.__matmul__(self, other)
        return LinearOperatorComposition(self, other)

    def __add__(self, other: Operator):
        """Operator addition."""
        if not isinstance(other, LinearOperator):
            return Operator.__add__(self, other)
        return LinearOperatorSum(self, other)

    def __mul__(self, other: torch.Tensor):
        """Operator multiplication with tensor."""
        return LinearOperatorElementwiseProduct(self, other)

    def __rmul__(self, other: torch.Tensor):
        """Operator multiplication with tensor."""
        return LinearOperatorElementwiseProduct(self, other)


class LinearOperatorComposition(LinearOperator, OperatorComposition):
    """Operator composition."""

    def adjoint(self, *args):
        """Adjoint operator composition."""
        return self._operator2.adjoint(self._operator1.adjoint(*args))


class LinearOperatorSum(LinearOperator, OperatorSum):
    """Operator addition."""

    def adjoint(self, *args):
        """Adjoint operator addition."""
        return self._operator1.adjoint(*args) + self._operator2.adjoint(*args)


class LinearOperatorElementwiseProduct(LinearOperator, OperatorElementwiseProduct):
    """Operator elementwise multiplication with scalar/tensor."""

    def adjoint(self, *args):
        """Adjoint Operator elementwise multiplication with scalar/tensor."""
        if self._tensor.is_complex():
            return self._operator.adjoint(*args) * self._tensor.conj()
        return self._operator.adjoint(*args) * self._tensor


class AdjointLinearOperator(LinearOperator):
    def __init__(self, operator: LinearOperator) -> None:
        super().__init__()
        self._operator = operator

    def forward(self, *args):
        """Adjoint of the operator."""
        return self._operator.adjoint(*args)

    def adjoint(self, *args):
        """Operator."""
        return self._operator.forward(*args)

    @property
    def H(self):
        """Adjoint of adjoint operator."""
        return self.operator
