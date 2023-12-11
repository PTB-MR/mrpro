"""General Operators."""

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
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVarTuple

import torch

Ts = TypeVarTuple('Ts')


class Operator(Generic[*Ts], ABC, torch.nn.Module):
    """The general Operator class."""

    @abstractmethod
    def forward(self, *args: *Ts): ...

    def __matmul__(self, other: Operator):
        """Operator composition."""
        return OperatorComposition(self, other)

    def __add__(self, other: Operator):
        """Operator addition."""
        return OperatorSum(self, other)

    def __mul__(self, other: torch.Tensor):
        """Operator multiplication with tensor."""
        return OperatorElementwiseProduct(self, other)

    def __rmul__(self, other: torch.Tensor):
        """Operator multiplication with tensor."""
        return OperatorElementwiseProduct(self, other)


class OperatorComposition(Operator):
    """Operator composition."""

    def __init__(self, operator1: Operator, operator2: Operator):
        super().__init__()
        self._operator1 = operator1
        self._operator2 = operator2

    def forward(self, *args):
        """Operator composition."""
        return self._operator1(self._operator2(*args))


class OperatorSum(Operator):
    """Operator addition."""

    def __init__(self, operator1: Operator, operator2: Operator):
        super().__init__()
        self._operator1 = operator1
        self._operator2 = operator2

    def forward(self, *args):
        """Operator addition."""
        return self._operator1(*args) + self._operator2(*args)


class OperatorElementwiseProduct(Operator):
    """Operator elementwise multiplication with scalar/tensor."""

    def __init__(self, operator: Operator, tensor: torch.Tensor):
        super().__init__()
        self._operator = operator
        self._tensor = tensor

    def forward(self, *args):
        """Operator elementwise multiplication."""
        return self._tensor * self._operator(*args)
