"""Class for Finite Difference Operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils.filters import filter_separable


class FiniteDifferenceOp(LinearOperator):
    r"""Finite Difference Operator.

    This pointwise operator, denoted as :math:`\nabla_h`, acts on a discrete image defined on
    a regular :math:`d`-dimensional grid and returns, at each pixel or voxel, a vector of finite differences of the
    image along a user-selected set of coordinate directions ``dim``
    (using forward, backward, or central finite difference stencils).

    Let :math:`u : \Omega_h \subset \mathbb{Z}^d \to \mathcal{K}` be a discrete image
    with values in :math:`\mathcal{K} \in \{\mathbb{R}, \mathbb{C}\}` and let ``dim`` :math:`\subset \{1,\dots,d\}`
    denote the set of active directions along which finite differences are computed.
    For a pixel or voxel :math:`x \in \Omega_h` and :math:`i \in` ``dim``, the forward finite difference
    (assuming unit spacing) in direction :math:`i` is

    .. math::

        (\nabla_h^{\mathrm{fwd}} u)_i(x) = u(x + e_i) - u(x),

    where :math:`e_i` is the unit vector in direction :math:`i`.
    Analogous formulas are used for the backward and central modes.

    In the continuous setting, for a function :math:`u : \Omega \subset \mathbb{R}^d \to \mathcal{K}`, the gradient is

    .. math::

        (\nabla u)_i(x) = \partial_{x_i} u(x),

    and :math:`\nabla_h` is the standard finite difference discretisation of :math:`\nabla`
    along the chosen directions ``dim``.

    As a simple 2D scalar example, an image :math:`u = u(x, y)` can be viewed as
    a function :math:`u : \mathbb{R}^2 \to \mathcal{K}` with

    .. math::

        \nabla u(x, y) = \bigl( \partial_x u(x, y), \partial_y u(x, y) \bigr),

    while :math:`\nabla_h u` replaces the partial derivatives by the corresponding finite differences
    in :math:`x` and :math:`y`. The operator implemented here applies this construction to the discrete image array
    and returns, at each pixel, the vector of directional differences along the active directions ``dim``.

    This finite difference gradient is used, for example, in :class:`mrpro.operators.SymmetrizedGradientOp`,
    where a discrete symmetrized gradient is formed from :math:`\nabla_h`.
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
