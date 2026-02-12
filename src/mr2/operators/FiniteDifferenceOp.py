"""Class for Finite Difference Operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils.filters import filter_separable


class FiniteDifferenceOp(LinearOperator):
    """Finite Difference Operator."""

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
