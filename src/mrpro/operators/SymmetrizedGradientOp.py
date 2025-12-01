"""Class for Symmetrized Gradient Operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.RearrangeOp import RearrangeOp


class SymmetrizedGradientOp(LinearOperator):
    r"""Discrete symmetrized gradient operator based on finite differences.

    This pointwise operator, denoted as :math:`\mathcal{E}_h`, acts on a discrete vector field defined on
    a regular :math:`d`-dimensional grid and returns, at each pixel or voxel, the symmetric part of
    the discrete gradient of the field (with respect to a a user-selected set of coordinate directions ``dim``).
    In other words, this operator takes, at each pixel, the matrix of directional finite differences of the
    vector field (along the active directions ``dim``) and symmetrizes it by averaging each pair of entries
    ``(i, j)`` and ``(j, i)``.

    Let :math:`v : \Omega_h \subset \mathbb{Z}^d \to \mathcal{K}^d` be a discrete vector field defined on
    an image grid :math:`\Omega_h` (equivalently, a stack of :math:`d` scalar images, for example
    the auxiliary field in TGV), with values in :math:`\mathcal{K} \in \{\mathbb{R}, \mathbb{C}\}`.
    Let :math:`\nabla_h v` denote the discrete gradient obtained by applying the finite difference operator
    :class:`mrpro.operators.FiniteDifferenceOp`.
    For each grid point :math:`x \in \Omega_h` the discrete symmetrized gradient is defined by

    .. math::

        \mathcal{E}_h v(x) = \frac{1}{2} \bigl( \nabla_h v(x) + \nabla_h v(x)^{\top} \bigr).

    Again, note that the operator acts pointwise, meaning that at each grid point :math:`x` it takes
    the local finite difference matrix :math:`\nabla_h v(x)` and replaces it by
    its symmetric part :math:`\frac{1}{2}(\nabla_h v(x) + \nabla_h v(x)^{\top})`.

    In the continuous setting, for a vector field :math:`v : \Omega \subset \mathbb{R}^d \to \mathcal{K}^d`,
    the symmetrized gradient :math:`\mathcal{E} v` is given by

    .. math::

        \mathcal{E} v(x) = \frac{1}{2} \bigl( \nabla v(x) + \nabla v(x)^{\top} \bigr),

    where :math:`\nabla v(x)` is the matrix of partial derivatives :math:`(\partial_{x_i} v_j(x))_{i,j=1}^d`.
    The discrete symmetrized gradient  operator :math:`\mathcal{E}_h` implemented here is the standard finite
    difference discretisation of :math:`\mathcal{E}`, obtained by replacing :math:`\nabla` with :math:`\nabla_h`.

    In 2D, for example, a vector field :math:`v(x, y) = (v_1(x, y), v_2(x, y))` has continuous symmetrized gradient

    .. math::

        \mathcal{E} v(x, y)
        = \frac{1}{2}
          \begin{pmatrix}
            \partial_x v_1 & \partial_y v_1 + \partial_x v_2 \\
            \partial_y v_1 + \partial_x v_2 & \partial_y v_2
          \end{pmatrix},

    while :math:`\mathcal{E}_h v` uses the corresponding finite differences in place of the partial derivatives.

    This discrete symmetrized gradient is used, for example, in Total Generalized Variation (TGV) regularization with
    the 2D case shown in [TGV]_.

    References
    ----------
    .. [TGV] Bredies, K. Recovering piecewise smooth multichannel images by minimization of convex
       functionals with total generalized variation penalty. In: Bruhn, A., Pock, T., Tai, X.C. (eds)
       Efficient Algorithms for Global Optimization Methods in Computer Vision. Lecture Notes in Computer Science,
       vol. 8293, Springer, Berlin, Heidelberg, 2014, pp. 44-77. https://doi.org/10.1007/978-3-642-54774-4_3
    """

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'backward',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ) -> None:
        """Symmetrized gradient operator.

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
        finite_difference_op = FiniteDifferenceOp(dim, mode=mode, pad_mode=pad_mode)
        transpose_op = RearrangeOp('sym_grad_dim  grad_dim  ...   ->   grad_dim  sym_grad_dim  ...')
        self._operator = 0.5 * (1 + transpose_op) @ finite_difference_op

    def __call__(self, v: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of symmetrized gradient operator.

        Since v is assumed to be the finite difference of a tensor d-dimensional tensor x (i.e., v = ∇x),
        the first dimension of v must match the number of dimensions specified in `dim` during initialization.

        Parameters
        ----------
        v
            d-dimensional input tensor

        Returns
        -------
        w
            A single-element tuple containing a (d+1)-dimensional tensor of the symmetrized gradient of v.
        """
        return super().__call__(v)

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of SymmetrizedGradientOp.

        .. note::
            Prefer calling the instance of the SymmetrizedGradientOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._operator(v)

    def adjoint(self, w: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply adjoint of symmetrized gradient operator.

        Parameters
        ----------
        w
            (d+1)-dimensional input tensor

        Returns
        -------
        v
            A single-element tuple containing the d-dimensional tensor of the adjoint of the symmetrized gradient.
        """
        return self._operator.adjoint(w)
