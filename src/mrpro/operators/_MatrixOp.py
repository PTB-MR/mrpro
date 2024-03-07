"""Fourier Operator."""

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

from mrpro.operators import LinearOperator


class MatrixOp(LinearOperator):
    """Simple Linear Operator that implements matrix multiplication."""

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.matrix = matrix

    def forward(self, x) -> tuple[torch.Tensor]:
        """Multiplication of input with matrix."""
        return (self.matrix @ x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Multiplication of input with adjoint/hermitian matrix."""
        return (self.matrix.mH @ x,)
