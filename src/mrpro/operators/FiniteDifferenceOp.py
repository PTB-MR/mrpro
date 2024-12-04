"""Class for Finite Difference Operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils.filters import filter_separable


class FiniteDifferenceOp(LinearOperator):
    """Finite Difference Operator."""

    @staticmethod
    def finite_difference_kernel(mode: str) -> torch.Tensor:
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
        ValueError
            If mode is not central, forward, backward or doublecentral
        """
        if mode == 'central':
            kernel = torch.tensor((-1, 0, 1)) / 2
        elif mode == 'forward':
            kernel = torch.tensor((0, -1, 1))
        elif mode == 'backward':
            kernel = torch.tensor((-1, 1, 0))
        else:
            raise ValueError(f'mode should be one of (central, forward, backward), not {mode}')
        return kernel

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'central',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ) -> None:
        """Finite difference operator.

        Parameters
        ----------
        dim
            Dimension along which finite differences are calculated.
        mode
            Type of finite difference operator
        pad_mode
            Padding to ensure output has the same size as the input
        """
        super().__init__()
        self.dim = dim
        self.pad_mode: Literal['constant', 'circular'] = 'constant' if pad_mode == 'zeros' else pad_mode
        self.register_buffer('kernel', self.finite_difference_kernel(mode))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward of finite differences.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Finite differences of x along dim stacked along first dimension
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
        """Adjoing of finite differences.

        Parameters
        ----------
        y
            Finite differences stacked along first dimension

        Returns
        -------
            Adjoint finite differences

        Raises
        ------
        ValueError
            If the first dimension of y is to the same as the number of dimensions along which the finite differences
            are calculated
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
