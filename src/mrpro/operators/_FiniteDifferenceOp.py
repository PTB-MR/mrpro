"""Class for Finite Difference Operator."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
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

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators._LinearOperator import LinearOperator
from mrpro.utils.filters import _filter_separable


class FiniteDifferenceOp(LinearOperator):
    """Finite Difference Operator.


    This work is inspired by:
    https://github.com/koflera/LearningRegularizationParameterMaps/blob/main/networks/grad_ops.py

    Please see https://github.com/koflera/LearningRegularizationParameterMaps/tree/main?tab=Apache-2.0-1-ov-file#readme
    for the copyright statement.

    """

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
        if mode == 'doublecentral':
            kernel = torch.tensor((-1, 0, 1))
        elif mode == 'central':
            kernel = torch.tensor((-1, 0, 1)) / 2
        elif mode == 'forward':
            kernel = torch.tensor((0, -1, 1))
        elif mode == 'backward':
            kernel = torch.tensor((-1, 1, 0))
        else:
            raise ValueError(f'mode should be one of (central, forward, backward, doublecentral), not {mode}')
        return kernel

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['doublecentral', 'central', 'forward', 'backward'] = 'doublecentral',
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
    ) -> None:
        """Finite difference operator.

        Parameters
        ----------
        dim
            Dimension along which finite differences are calculated.
        mode
            Type of finite difference operator
        padding_mode
            Padding to ensure output has the same size as the input
        """
        super().__init__()
        self.dim = dim
        self.padding_mode = padding_mode
        self.kernel = self.finite_difference_kernel(mode)
        self.adjoint_kernel = torch.flip(self.kernel, dims=(-1,))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward of finite differences.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Finite differences of x along dim stacked along first dimension.
        """
        return (torch.stack([_filter_separable(x, (self.kernel,), axis=(d,)) for d in self.dim]),)

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
                    [_filter_separable(y[i, ...], (self.adjoint_kernel,), axis=(d,)) for i, d in enumerate(self.dim)]
                ),
                dim=0,
            ),
        )
