"""Primal-Dual Hybrid Gradient Algorithm (PDHG)."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Self, TypeVar, overload

import numpy as np
import torch

from mrpro.algorithms.optimizers import OptimizerStatus
from mrpro.operators import IdentityOp, LinearOperator, Operator, ProximableFunctional


@dataclass
class PDHGStatus(OptimizerStatus):
    """Status of the PDHG algorithm."""

    objective: Callable[[*tuple[torch.Tensor, ...]], torch.Tensor]
    dual_stepsize: float | torch.Tensor
    primal_stepsize: float | torch.Tensor
    relaxation: float | torch.Tensor
    duals: Sequence[torch.Tensor]
    x_relaxed: Sequence[torch.Tensor]

class LinearOperatorMatrix(Operator):
    """A matrix of linear operators."""

    def __init__(self, operators: Sequence[Sequence[LinearOperator]]):
        super().__init__()
        try:
            self._operators = np.array(operators)
        except ValueError as e:
            raise ValueError('operators must be a 2D array') from e
        if self._operators.ndim != 2:
            raise ValueError('operators must be a 2D array')
        if not all(isinstance(op, LinearOperator) for op in self._operators.flat):
            raise ValueError('all elements of operators must be LinearOperators')


    @property
    def shape(self) -> tuple[int, int]:
        return self._operators.shape

    @property
    def __getitem__(self, idx:Any) -> LinearOperator:
        new_ops = self._operators[idx]
        return Self(new_ops)

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        return iter(self._rows)


    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(x) != self.shape[1]
            raise ValueError('number of input tensors must match number of columns in operator')
        ret = (sum(op(xi)[0] for op, xi in zip(row, x, strict=True)) for row in self._rows)

    @property
    def H(self) -> LinearOperatorMatrix:
        return LinearOperatorMatrix(*self._op.H)

    def operator_norm(self, initial_values: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        return self._op.operator_norm(initial_values, **kwargs)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.H(x)

class ZeroFunctional(ProximableFunctional):
    """The constant zero functional."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the functional to the tensor.

        Always returns 0.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
        Result of the functional applied to x.
        """
        return (x.new_zeros(1),)

    def prox(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:  # noqa ARG002
        """Apply the proximal operator to a tensor.

        Always returns x.

        Parameters
        ----------
        x
            input tensor
        sigma
            step size

        Returns
        -------
            Result of the proximal operator applied to x
        """
        return (x,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:  # noqa ARG002
        """Apply the proximal operator of the convex conjugate of the functional to a tensor.

        Always returns x.

        Parameters
        ----------
        x
            input tensor
        sigma
            step size

        Returns
        -------
            Result of the proximal operator of the convex conjugate applied to x
        """
        return (x,)


T = TypeVar('T', bound=Operator, covariant=True)


class _OperatorHStack(Operator, Generic[T]):
    """A generalization of horizontal stacking of operators."""

    def __init__(self, *operators: T):
        super().__init__()
        self.operators = torch.nn.ModuleList(operators)

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor]:
        ys = [op(xi) for op, xi in zip(self.operators, x, strict=False)]
        return tuple(sum(el) for el in zip(*ys, strict=False))

    def __len__(self):
        return len(self.operators)

    def __iter__(self):
        return iter(self.operators)


class _OperatorVStack(Operator, Generic[T]):
    def __init__(self, *operators: T):
        super().__init__()
        self.operators = torch.nn.ModuleList(operators)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the operator to the tensor."""
        return sum((op(x) for op in self.operators), start=())


LinearOperatorOrHStack = TypeVar('LinearOperatorOrHStack', LinearOperator, '_LinearOperatorHStack')
LinearOperatorOrVStack = TypeVar('LinearOperatorOrVStack', LinearOperator, '_LinearOperatorVStack')


class _LinearOperatorHStack(_OperatorHStack[LinearOperatorOrVStack]):
    @overload
    def H(self: _LinearOperatorHStack[LinearOperator]) -> _LinearOperatorVStack[LinearOperator]: ...
    @overload
    def H(
        self: _LinearOperatorHStack[_OperatorVStack[LinearOperator]],
    ) -> _LinearOperatorVStack[_LinearOperatorHStack[LinearOperator]]: ...

    @property
    def H(  # noqa: N802
        self: _OperatorHStack[LinearOperator | _OperatorVStack[LinearOperator]],
    ) -> _OperatorVStack[LinearOperator | _OperatorHStack[LinearOperator]]:
        """Return the adjoint of the operator."""
        return _OperatorVStack(*(op.H for op in self.operators))

    def operator_norm(
        self,
        initial_values: Sequence[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Estimate the operator norm of the operator, see :meth:`LinearOperator.operator_norm`.

        |[A B]|_op = |A|_op + |B|_op
        """
        zero = initial_values[0].new_zeros(1)
        norm = sum(
            (operator.operator_norm(x, **kwargs) for operator, x in zip(self.operators, initial_values, strict=False)),
            start=zero,
        )
        return norm


class _ProximalFunctionalSeparableSum(_OperatorHStack[ProximableFunctional]):
    def prox(self, *x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, ...]:
        """Apply the proximal operator to the tensor."""
        return tuple(f.prox(xi, sigma)[0] for f, xi in zip(self.operators, x, strict=False))

    def prox_convex_conj(self, *x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, ...]:
        """Apply the proximal operator of the convex conjugate of the functional to a tensor."""
        return tuple(f.prox_convex_conj(xi, sigma)[0] for f, xi in zip(self.operators, x, strict=False))


class _LinearOperatorVStack(_OperatorVStack[LinearOperatorOrHStack]):
    @overload
    def H(self: _LinearOperatorVStack[LinearOperator]) -> _LinearOperatorHStack[LinearOperator]: ...
    @overload
    def H(
        self: _LinearOperatorVStack[_OperatorHStack[LinearOperator]],
    ) -> _LinearOperatorHStack[_LinearOperatorVStack[LinearOperator]]: ...
    @property
    def H(  # noqa: N802
        self,
    ) -> _OperatorHStack[LinearOperator | _OperatorVStack[LinearOperator]]:
        """Return the adjoint of the operator."""
        return _OperatorHStack(*(op.H for op in self.operators))

    @overload
    def operator_norm(
        self: _OperatorVStack[LinearOperator], initial_values: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...

    @overload
    def operator_norm(
        self: _OperatorVStack[_OperatorHStack[LinearOperator]], initial_values: Sequence[torch.Tensor], **kwargs
    ) -> torch.Tensor: ...

    def operator_norm(
        self: _OperatorVStack[LinearOperator | _OperatorHStack[LinearOperator]],
        initial_values: torch.Tensor | Sequence[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Estimate the operator norm of the operator, see :meth:`LinearOperator.operator_norm`.

        |[A, B]^T|_op = sqrt(|A|_op^2 + |B|_op^2)

        """
        norm = sum(operator.operator_norm(initial_values, **kwargs) ** 2 for operator in self.operators) ** 0.5
        return norm

    def __len__(self):
        return len(self.operators)

    def __iter__(self):
        return iter(self.operators)


class _LinearOperatorMatrix(_LinearOperatorVStack[_LinearOperatorHStack[LinearOperator]]):
    def __init__(self, operators: Sequence[Sequence[LinearOperator]]):
        super().__init__()
        self._op = _LinearOperatorVStack(*(_LinearOperatorHStack(*row) for row in operators))

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self._op(*x)

    @property
    def H(self) -> _LinearOperatorMatrix:
        return _LinearOperatorMatrix(*self._op.H)

    def operator_norm(self, initial_values: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        return self._op.operator_norm(initial_values, **kwargs)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.H(x)


def pdhg(
    x0: Sequence[torch.Tensor],
    f: Sequence[ProximableFunctional] | None = None,
    g: Sequence[ProximableFunctional] | None = None,
    operator: Sequence[Sequence[LinearOperator]] | None = None,
    n_iterations: int = 10,
    primal_stepsize: float | None = None,
    dual_stepsize: float | None = None,
    relaxation: float = 1.0,
    x_relaxed: Sequence[torch.Tensor] | None = None,
    duals: Sequence[torch.Tensor] | None = None,
    callback: Callable[[PDHGStatus], None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Primal-Dual Hybrid Gradient Algorithm (PDHG).

    Solves the minimization problem
        :math:`\min_x g(x) + f(A x)`
    with linear operator A and proximable functionals f and g.

    The operator is supplied as a matrix (tuple of tuples) of linear operators,
    f and g are supplied as tuples of proximable functionals interpreted as separable sums.

    Thus, problem solved is
            :math:`\min_x \sum_i,j g_j(x_j) + f_i(A_ij x_j)`.

    If neither primal nor dual step size are not supplied, they are chose as 1/|A|_op.
    If either is supplied, the other is chosen such that primal_stepsize*dual_stepsize = 1/|A|_op^2

    For a warm start, the relaxed solution x_relaxed and dual variables can be supplied.
    These might be obtained from the Status object of a previous run.

    Parameters
    ----------
    x0
        initial guess
    f
        tuple of proximable functionals interpreted as a separable sum

    g
        tuple of proximable functionals interpreted as a separable sum
    operator
        matrix of linear operators
    n_iterations
        number of iterations
    dual_stepsize
        dual step size
    primal_stepsize
        primal step size
    relaxtion
        relaxation parameter
    x_relaxed
        relaxed solution, used for warm start
    relaxation
        relaxation parameter, 1.0 is no relaxation
    duals
        dual variables, used for warm start
    callback
        callback function called after each iteration


    """
    if f is None and g is None:
        warnings.warn(
            'Both f and g are None. The objective is constant. Returning x0 as a possible solution', stacklevel=2
        )
        return tuple(x0)

    if operator is None:
        rows = len(f) if f is not None else 1
        cols = len(g) if g is not None else 1
    else:
        rows = len(operator)
        all_cols = [len(row) for row in operator]
        if len(set(all_cols)) != 1:
            raise ValueError('operator must have same number of columns in each row')
        cols = all_cols[0]
        if f is not None and len(f) != rows:
            raise ValueError('f must have same length as operator has number of rows')
        if g is not None and len(g) != cols:
            raise ValueError('g must have same length as operator has number of columns')

    f_ = _OperatorHStack(*((ZeroFunctional(),) * rows if f is None else f))
    g_ = _OperatorHStack(*((ZeroFunctional(),) * cols if g is None else g))
    operator_ = _LinearOperatorVStack(
        *[_LinearOperatorHStack(*row) for row in (((IdentityOp(),) * cols,) * rows if operator is None else operator)]
    )

    if primal_stepsize is None or dual_stepsize is None:
        # choose primal and dual step size such that their product is 1/|operator|**2
        # to ensure convergence
        operator_norm = operator_.operator_norm(x0)
        if primal_stepsize is None and dual_stepsize is None:
            primal_stepsize_ = dual_stepsize = 1.0 / operator_norm
        elif primal_stepsize is None:
            primal_stepsize_ = 1 / (operator_norm * dual_stepsize)
        elif dual_stepsize is None:
            dual_stepsize_ = 1 / (operator_norm * primal_stepsize)
    else:
        primal_stepsize_ = primal_stepsize
        dual_stepsize_ = dual_stepsize

    primals_relaxed = x0 if x_relaxed is None else x_relaxed
    duals_ = [0.0 * operator_(x0)] if duals is None else duals

    if len(duals_) != rows:
        raise ValueError('if dual y is supplied, it should be a tuple of same length as the tuple of g')

    primals_ = x0
    for i in range(n_iterations):
        duals_ = tuple(
            dual + dual_stepsize_ * step for dual, step in zip(duals_, operator_(primals_relaxed), strict=False)
        )
        duals_ = f_.prox_convex_conj(*duals_, sigma=dual_stepsize_)

        primals_new = tuple(
            primal - primal_stepsize_ * step for primal, step in zip(primals_, operator_.H(duals_), strict=False)
        )
        primals_new = g_.prox(*primals_new, sigma=primal_stepsize_)
        primals_relaxed = [
            torch.lerp(primal, primal_new, relaxation)
            for primal, primal_new in zip(primals_, primals_new, strict=False)
        ]
        if callback is not None:
            status = PDHGStatus(
                iteration_number=i,
                dual_stepsize=dual_stepsize_,
                primal_stepsize=primal_stepsize_,
                relaxation=relaxation,
                duals=duals_,
                solution=tuple(primals_),
                x_relaxed=primals_relaxed,
                objective=lambda *x: f_.forward(*operator_(*x))[0] + g_.forward(*x)[0],
            )
            callback(status)
    return tuple(primals_)
