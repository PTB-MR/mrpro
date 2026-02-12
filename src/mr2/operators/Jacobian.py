"""Jacobian."""

from collections.abc import Callable

import torch

from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.Operator import Operator


class Jacobian(LinearOperator):
    """Jacobian of an Operator.

    This operator implements the Jacobian of a (non-linear) operator at a given point :math:`x_0` as a LinearOperator,
    i.e. a linearization of the operator at the point :math:`x_0`.
    """

    def __init__(self, operator: Operator[torch.Tensor, tuple[torch.Tensor]], *x0: torch.Tensor):
        """Initialize the Jacobian operator.

        Parameters
        ----------
        operator
            operator to linearize
        x0
            point at which to linearize the operator
        """
        super().__init__()
        self._vjp: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] | None = None
        self._x0: tuple[torch.Tensor, ...] = x0
        self._operator = operator
        self._f_x0: tuple[torch.Tensor, ...] | None = None

    def adjoint(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # type:ignore[override]
        """Apply the adjoint operator.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            output tensor
        """
        if self._vjp is None:
            self._f_x0, self._vjp = torch.func.vjp(self._operator, *self._x0)
        assert self._vjp is not None  # noqa: S101 (hint for mypy)
        return (self._vjp(x)[0],)

    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # type:ignore[override]
        """Apply the operator.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            output tensor
        """
        return super().__call__(*x)

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # type:ignore[override]
        """Apply the operator.

        .. note::
            Prefer calling the instance of the Jacobian as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        self._f_x0, jvp = torch.func.jvp(self._operator, self._x0, x)
        return jvp

    @property
    def value_at_x0(self) -> tuple[torch.Tensor, ...]:
        """Evaluation of the operator at the point :math:`x_0`."""
        if self._f_x0 is None:
            self._f_x0 = self._operator(*self._x0)
        assert self._f_x0 is not None  # noqa: S101 (hint for mypy)
        return self._f_x0

    def taylor(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Taylor approximation of the operator.

        Approximate the operator at x by a first order Taylor expansion around :math:`x_0`,
        :math:`f(x_0) + J_f(x_0)(x - x_0)`.

        This is not faster than the forward method of the operator itself, as the calculation of the
        jacobian-vector-product requires the forward pass of the operator to be computed.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            Value of the Taylor approximation at x
        """
        delta = tuple(ix - ix0 for ix, ix0 in zip(x, self._x0, strict=False))
        self._f_x0, jvp = torch.func.jvp(self._operator, self._x0, delta)
        assert self._f_x0 is not None  # noqa: S101 (hint for mypy)
        f_x = tuple(ifx0 + ijvp for ifx0, ijvp in zip(self._f_x0, jvp, strict=False))
        return f_x

    def gauss_newton(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Calculate the Gauss-Newton approximation of the Hessian of the operator.

        Returns :math:`J^T J x`, where :math:`J` is the Jacobian of the operator at :math:`x_0`.
        Uses backward and forward automatic differentiation of the operator.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            Gauss-Newton approximation of the Hessian applied to x
        """
        return self.adjoint(*self(*x))
