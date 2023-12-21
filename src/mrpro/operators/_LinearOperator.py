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


class LinearOperator(Operator):
    """General Linear Operator."""

    @abstractmethod
    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def H(self):
        """Adjoint operator."""
        return AdjointLinearOperator(self)


class AdjointLinearOperator(LinearOperator):
    def __init__(self, operator: LinearOperator) -> None:
        super().__init__()
        self._operator = operator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adjoint of the operator."""
        return self._operator.adjoint(x)

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        """Operator."""
        return self._operator.forward(x)

    @property
    def H(self):
        """Adjoint of adjoint operator."""
        return self.operator
