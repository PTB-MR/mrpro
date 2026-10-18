"""Class for Finite Difference Operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils.filters import filter_separable


class FiniteDifferenceOp(LinearOperator):
    r"""Finite difference operator.

    This pointwise operator computes finite differences of a discrete :math:`d`-dimensional tensor ``x``.
    Differences are computed along the axes listed in ``dim``
    (e.g. ``dim=(-2, -1)`` for the last two axes)
    by means of a separable convolution with appropriate filters
    (supported modes are ``forward``, ``backward``, and ``central``).
    The output is a :math:`(d+1)`-dimensional tensor ``y``
    (``y.shape[0] == len(dim)``)
    where each ``y[i]`` is the finite difference tensor along the selected axis ``dim[i]``.

    For example, the forward finite difference ``nabla(x)`` along a chosen axis ``dim[i]`` can be written as

    .. code-block:: python

        y[i, *k] = nabla(x)[i, *k] = x[*(k + e_i)] - x[*k]

    for every coordinate ``k = (k_1, ..., k_d)`` in the grid.
    Here ``e_i = (e_i_1, ..., e_i_d)`` is the unit vector in direction ``dim[i]``,
    i.e. ``e_i[j] = 1`` if ``j == dim[i]`` else ``0``.

    Boundary handling (e.g. when coordinate ``k + e_i`` is outside the grid) is controlled by ``pad_mode``.
    """

    @staticmethod
    def finite_difference_kernel(mode: Literal['central', 'forward', 'backward']) -> torch.Tensor:
        """Finite difference kernel.

        Parameters
        ----------
        mode
            String specifying kernel type

        Returns
        -------
            Finite difference kernel

        Raises
        ------
        `ValueError`
            If mode is not forward, backward or central
        """
        if mode == 'forward':
            kernel = torch.tensor((0, -1, 1))
        elif mode == 'backward':
            kernel = torch.tensor((-1, 1, 0))
        elif mode == 'central':
            kernel = torch.tensor((-1, 0, 1)) / 2
        else:
            raise ValueError(f'mode should be one of (central, forward, backward), not {mode}')
        return kernel

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'forward',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ) -> None:
        """Finite difference operator.

        Parameters
        ----------
        dim
            Dimensions along which finite differences are calculated.
        mode
            Type of finite difference operator
        pad_mode
            Padding to ensure output has the same size as the input
        """
        super().__init__()
        self.dim = dim
        self.pad_mode: Literal['constant', 'circular'] = 'constant' if pad_mode == 'zeros' else pad_mode
        self.kernel = self.finite_difference_kernel(mode)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward finite difference operation.

        Calculates finite differences of the input tensor `x` along the dimensions
        specified during initialization. The results for each dimension are stacked
        along the first dimension of the output tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Tensor containing the finite differences of `x` along the specified
            dimensions, stacked along the first dimension.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of FiniteDifferenceOp.

        .. note::
            Prefer calling the instance of the FiniteDifferenceOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return (
            torch.stack(
                [
                    filter_separable(x, (self.kernel,), dim=(dim,), pad_mode=self.pad_mode, pad_value=0.0)
                    for dim in self.dim
                ]
            ),
        )

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply adjoint finite difference operation.

        This operation is the adjoint of the forward finite difference calculation.
        It takes a tensor `y` (which is assumed to be the output of the forward pass,
        i.e., finite differences stacked along the first dimension) and computes
        the sum of the adjoints of the individual directional finite difference operations.

        Parameters
        ----------
        y
            Input tensor, representing finite differences stacked along the first dimension.
            The size of the first dimension must match the number of dimensions
            specified for the operator.

        Returns
        -------
            Result of the adjoint finite difference operation.

        Raises
        ------
        ValueError
            If the first dimension of `y` does not match the number of
            dimensions along which finite differences were calculated.
        """
        if y.shape[0] != len(self.dim):
            raise ValueError('Fist dimension of input tensor has to match the number of finite difference directions.')
        return (
            torch.sum(
                torch.stack(
                    [
                        filter_separable(
                            yi,
                            (torch.flip(self.kernel, dims=(-1,)),),
                            dim=(dim,),
                            pad_mode=self.pad_mode,
                            pad_value=0.0,
                        )
                        for dim, yi in zip(self.dim, y, strict=False)
                    ]
                ),
                dim=0,
            ),
        )
