"""Jacobian."""

from collections.abc import Callable

import torch

from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.Operator import Operator


class Jacobian(LinearOperator):
    """Jacobian of an Operator.

    This operator implements the Jacobian of a (non-linear) operator at a given point x0 as a LinearOperator, i.e. a linearization of the operator at the point `x0.`
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

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # type:ignore[override]
        """Apply the operator.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            output tensor
        """
        self._f_x0, jvp = torch.func.jvp(self._operator, self._x0, x)
        return jvp

    @property
    def value_at_x0(self) -> tuple[torch.Tensor, ...]:
        """Evaluation of the operator at the point x0."""
        if self._f_x0 is None:
            self._f_x0 = self._operator(*self._x0)
        assert self._f_x0 is not None  # noqa: S101 (hint for mypy)
        return self._f_x0

    def taylor(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Taylor approximation of the operator.

        Approximate the operator at x by a first order Taylor expansion around x0.

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
# compute f(x) = f(x_0) + J_f(x_0)(x - x_0)
f_x = tuple(ifx0 + ijvp for ifx0, ijvp in zip(self._f_x0, jvp, strict=False))
        return f_x

    def gauss_newton(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Calculate the Gauss-Newton approximation of the Hessian of the operator.

        Returns J^T J x, where J is the Jacobian of the operator at x0.
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
