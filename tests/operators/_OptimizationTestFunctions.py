"""Test functions for non-linear optimization."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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


import torch

from mrpro.operators import Operator


# TODO: Consider introducing the concept of a "Functional" for scalar-valued operators
class Rosenbrock(Operator):
    def __init__(self, a: float = 1, b: float = 100) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor,]:
        fval = (self.a - x1) ** 2 + self.b * (x1 - x2**2) ** 2

        return (fval,)


class Booth(Operator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor,]:
        fval = (x1 + 2.0 * x2 - 7.0) ** 2 + (2.0 * x1 - +x2 - 5.0) ** 2

        return (fval,)
