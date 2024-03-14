"""Generalized Sum Multiplication Operator."""

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

import re

import torch

from mrpro.operators import LinearOperator


class EinsumOp(LinearOperator):
    """A Linear Operator that implements sum products in einstein notation.

    Implements A_{indices_A}*x^{indices_x} = y{indices_y}
    with Einstein summation rules over the indices, see torch.einsum for more information.


    It can be used to implement tensor contractions, such as for example, different versions of
    matrix-vector / matrix-matrix products of the form A @ x, depending on the chosen einsum rules and
    shapes of A and x.

    Examples are:
    - matrix-vector multiplication of A and the batched vector x = [x1,...,xN] consisting
      of N vectors x1, x2, ..., xN. Then, the operation defined by
        A @ x := diag(A, A, ..., A) * [x1, x2, ..., xN]^T = [A*x1, A*x2, ..., A*xN]^T
      can be implemented by the einsum rule
        "...ij,j->...i"

    - matrix-vector multiplication of a matrix A consisting of N different matrices
      A1, A2, ... AN with one vector x. Then, the operation defined by
        A @ x: = diag(A1, A2,..., AN) * [x, x, ..., x]^T
      can be implemented by the einsum rule
        "...ij,j->...i"

    - matrix-vector multiplication of a matrix A consisting of N different matrices
      A1, A2, ... AN with a vector x = [x1,...,xN] consisting
      of N vectors x1, x2, ..., xN. Then, the operation defined by
        A @ x: = diag(A1, A2,..., AN) * [x1, x2, ..., xN]^T
      can be implemented by the einsum rule
        "...ij,j->...i""...ij,...j->...i"
      This is the default behaviour of the operator.
    """

    def __init__(self, matrix: torch.Tensor, einsum_rule: str = '...ij,...j->...i') -> None:
        """Initialize Einsum Operator.

        Parameters
        ----------
        matrix
            'Matrix' `A` to be used as first factor in the sum product A*x

        einsum_rule
            Einstein summation rule describing the forward of the operator.
            Also see torch.einsum for more information.
        """
        super().__init__()
        if (match := re.match('(.+),(.+)->(.+)', einsum_rule)) is None:
            raise ValueError(f'Einsum pattern should match (.+),(.+)->(.+) but got {einsum_rule}.')
        indices_matrix, indices_input, indices_output = match.groups()
        # swapping the input and output indices gets the adjoint rule
        self._adjoint_rule = f'{indices_matrix},{indices_output}->{indices_input}'
        self._forward_rule = einsum_rule
        self.matrix = torch.nn.Parameter(matrix, matrix.requires_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Sum-Multiplication of input `x` with `A`.

        `A` and the rule used to perform the sum-product is set at initialization.

        Parameters
        ----------
        x
            input tensor to be multiplied with the 'matrix' A

        Returns
        -------
            result of matrix-vector multiplication
        """
        y = torch.einsum(self._forward_rule, self.matrix, x)
        return (y,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Multiplication of input with the adjoint of `A`.

        Parameters
        ----------
        y
            tensor to be multiplied with hermitian/adjoint 'matrix' A

        Returns
        -------
            result of adjoint sum product
        """
        x = torch.einsum(self._adjoint_rule, self.matrix.conj(), y)
        return (x,)
