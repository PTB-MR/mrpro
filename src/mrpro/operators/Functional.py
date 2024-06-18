"""Base Class Functional."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence

import torch

from mrpro.operators._Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class."""

    def __init__(
        self, weight: torch.Tensor | float = 1.0, target: torch.Tensor | None = None, dim: Sequence[int] | None = None
    ) -> None:
        """Initialize a Functional.

        Parameters
        ----------
            weight
                weighting of the norm
            target
                element to which distance is taken - often data tensor
            dim
                dimension over which norm is calculated
        """
        super().__init__()
        self.register_buffer('weight', torch.as_tensor(weight))
        if target is None:
            target = torch.tensor([0.0], dtype=torch.float32)
        self.register_buffer('target', target)
        self.dim = dim


class ProximableFunctional(Functional, ABC):
    """ProximableFunction Base Class."""

    @abstractmethod
    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply proximal operator."""

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply proximal of convex conjugate of functional."""
        return (x - sigma * self.prox(x * 1 / sigma, 1 / sigma),)
