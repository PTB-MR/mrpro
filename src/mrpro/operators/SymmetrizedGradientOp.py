"""Symmetrized gradient operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.RearrangeOp import RearrangeOp


class SymmetrizedGradientOp(LinearOperator):
    r"""Symmetrized gradient operator.

    The symmetrized gradient :math:`\mathcal{E}: \mathcal{R}^d \to \mathcal{R}^{d+1}`
    is defined as:

    .. math::

        \mathcal{E}v = \frac{1}{2}(\nabla v + (\nabla v)^{\top})

    where :math:`(\nabla v)^{\top}` denotes the transpose of the Jacobian matrix :math:`\nabla v`.

    This is used, for example, in Total Generalized Variation (TGV) regularization with the 2D case
    shown in [TGV]_.

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
        mode: Literal['central', 'forward', 'backward'] = 'central',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ) -> None:
        """Symmetrized gradient operator.

        Parameters
        ----------
        dim
            Dimension along which finite differences are calculated.
            k elements for k gradients.
        mode
            Type of finite difference operator
        pad_mode
            Padding to ensure output has the same size as the input
        """
        super().__init__()
        finite_difference_op = FiniteDifferenceOp(dim, mode=mode, pad_mode=pad_mode)
        transpose_op = RearrangeOp('sym_grad_dim  grad_dim  ...   ->   grad_dim  sym_grad_dim  ...')
        self.symmetric_gradient_op = 0.5 * (1 + transpose_op) @ finite_difference_op

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
        return self.symmetric_gradient_op(v)

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
        return self.symmetric_gradient_op.adjoint(w)
