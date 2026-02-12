"""Generalized Sum Multiplication Operator."""

import re

import torch
from einops import einsum

from mr2.operators.LinearOperator import LinearOperator


class EinsumOp(LinearOperator):
    r"""A Linear Operator that implements sum products in Einstein notation.

    Implements :math:`A_{\mathrm{indices}_A}*x^{\mathrm{indices}_x} = y_{\mathrm{indices}_y}`
    with Einstein summation rules over the :math:`indices`, see `torch.einsum` or `einops.einsum`
    for more information. Note, that the indices must be space separated (einops convention).


    It can be used to implement tensor contractions, such as for example, different versions of
    matrix-vector or matrix-matrix products of the form `A @ x`, depending on the chosen einsum rules and
    shapes of `A` and `x`.

    Examples are:

    - matrix-vector multiplication of :math:`A` and the batched vector :math:`x = [x_1, ..., x_N]` consisting
      of :math:`N` vectors :math:`x_1, x_2, ..., x_N`. Then, the operation defined by
      :math:`A @ x := \mathrm{diag}(A, A, ..., A) [x_1, x_2, ..., x_N]^T` = :math:`[A x_1, A x_2, ..., A x_N]^T`
      can be implemented by the einsum rule ``'i j, ... j -> ... i'``.

    - matrix-vector multiplication of a matrix :math:`A` consisting of :math:`N` different matrices
      :math:`A_1, A_2, ... A_N` with one vector :math:`x`. Then, the operation defined by
      :math:`A @ x := \mathrm{diag}(A_1, A_2,..., A_N) [x, x, ..., x]^T`
      can be implemented by the einsum rule ``'... i j, j -> ... i'``.

    - matrix-vector multiplication of a matrix :math:`A` consisting of :math:`N` different matrices
      :math:`A_1, A_2, ... A_N` with a vector :math:`x = [x_1,...,x_N]` consisting
      of :math:`N` vectors :math:`x_1, x_2, ..., x_N`. Then, the operation defined by
      :math:`A @ x := \mathrm{diag}(A_1, A_2,..., A_N) [x_1, x_2, ..., x_N]^T`
      can be implemented by the einsum rule ``'... i j, ... j -> ... i'``.
      This is the default behavior of the operator.
    """

    def __init__(self, matrix: torch.Tensor, einsum_rule: str = '... i j, ... j -> ... i') -> None:
        """Initialize Einsum Operator.

        Parameters
        ----------
        matrix
            Matrix :math:`A` to be used as first factor in the sum product :math:`A*x`

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

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply sum-product of input `x` with the operator's matrix `A`.

        :math:`A` and the rule used to perform the sum-product is set at initialization.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Result of the sum-product operation.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of EinsumOp.

        .. note::
            Prefer calling the instance of the EinsumOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        y = einsum(self.matrix, x, self._forward_pattern)
        return (y,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Multiplication of input with the adjoint of :math:`A`.

        Parameters
        ----------
        y
            Tensor to be multiplied with hermitian/adjoint matrix :math:`A`

        Returns
        -------
            Result of the adjoint sum-product operation.
        """
        x = einsum(self.matrix.conj(), y, self._adjoint_pattern)
        return (x,)
