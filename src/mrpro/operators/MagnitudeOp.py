"""Operator returning the magnitude of the input."""

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

from typing import TypeVar
from typing import TypeVarTuple

import torch

from mrpro.operators.EndomorphOperator import EndomorphOperator
from mrpro.operators.EndomorphOperator import endomorph

Tin = TypeVarTuple('Tin')  # TODO: bind to torch.Tensors
Tout = TypeVar('Tout', bound=tuple, covariant=True)  # TODO: bind to torch.Tensor


class MagnitudeOp(EndomorphOperator):
    """Magnitude of input tensors."""

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # def forward(self, *x):
        """Magnitude of tensors.

        Parameters
        ----------
        x
            input tensors

        Returns
        -------
            tensors with magnitude (absolute values) of input tensors
        """
        return tuple([torch.abs(xi) for xi in x])
