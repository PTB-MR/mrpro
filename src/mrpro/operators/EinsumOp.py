"""Generalized Sum Multiplication Operator."""

import re

import torch
from einops import einsum

from mrpro.operators.LinearOperator import LinearOperator


class EinsumOp(LinearOperator):
    r"""A Linear Operator that implements sum products in Einstein notation.

    Implements :math:`A_{indices_A}*x^{indices_x} = y_{indices_y}`
    with Einstein summation rules over the indices, see torch.einsum or einops.einsum
    for more information. Note, that the indices must be space separated (einops convention).


    It can be used to implement tensor contractions, such as for example, different versions of
    matrix-vector or matrix-matrix products of the form :math:`A @ x`, depending on the chosen einsum rules and
    shapes of :math:`A` and :math:`x`.

    Examples are:

    - matrix-vector multiplication of :math:`A` and the batched vector :math:`x = [x1,...,xN]` consisting
      of :math:`N` vectors :math:`x1, x2, ..., xN`. Then, the operation defined by
      :math:`A @ x := diag(A, A, ..., A) * [x1, x2, ..., xN]^T = [A*x1, A*x2, ..., A*xN]^T`
      can be implemented by the einsum rule ``"i j, ... j -> ... i"``.

    - matrix-vector multiplication of a matrix :math:`A` consisting of :math:`N` different matrices
      :math:`A1, A2, ... AN` with one vector :math:`x`. Then, the operation defined by
      :math:`A @ x: = diag(A1, A2,..., AN) * [x, x, ..., x]^T`
      can be implemented by the einsum rule ``"... i j, j -> ... i"``.

    - matrix-vector multiplication of a matrix :math:`A` consisting of :math:`N` different matrices
      :math:`A1, A2, ... AN` with a vector :math:`x = [x1,...,xN]` consisting
      of :math:`N` vectors :math:`x1, x2, ..., xN`. Then, the operation defined by
      :math:`A @ x: = diag(A1, A2,..., AN) * [x1, x2, ..., xN]^T`
      can be implemented by the einsum rule ``"... i j, ... j -> ... i"``.
      This is the default behavior of the operator.
    """

    def __init__(self, matrix: torch.Tensor, einsum_rule: str = '... i j, ... j -> ... i') -> None:
        """Initialize Einsum Operator.

        Parameters
        ----------
        matrix
            'Matrix' :math:`A` to be used as first factor in the sum product :math:`A*x`

        einsum_rule
            Einstein summation rule describing the forward of the operator.
            Also see torch.einsum for more information.
        """
        super().__init__()
        if (match := re.match('(.+),(.+)->(.+)', einsum_rule)) is None:
            raise ValueError(f'Einsum pattern should match (.+),(.+)->(.+) but got {einsum_rule}.')
        indices_matrix, indices_input, indices_output = match.groups()
        # swapping the input and output indices gets the adjoint rule
        self._adjoint_pattern = f'{indices_matrix},{indices_output}->{indices_input}'
        self._forward_pattern = einsum_rule
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
        y = einsum(self.matrix, x, self._forward_pattern)
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
        x = einsum(self.matrix.conj(), y, self._adjoint_pattern)
        return (x,)
