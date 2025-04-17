"""Conjugate Gradient for linear systems with self-adjoint positive semidefinite linear operator."""

from collections.abc import Callable, Sequence

import torch
from typing_extensions import Unpack, overload

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.LinearOperatorMatrix import LinearOperatorMatrix


def vdot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    """Vector dot product."""
    return sum(
        (torch.vdot(a_i.flatten(), b_i.flatten()) for a_i, b_i in zip(a, b, strict=True)), start=torch.tensor(0.0)
    )


class CGStatus(OptimizerStatus):
    """Status of the conjugate gradient algorithm."""

    residual: tuple[torch.Tensor, ...]
    """Residual of the current estimate."""

    preconditioned_residual: tuple[torch.Tensor, ...]
    """Preconditioned residual of the current estimate."""


OperatorMatrixLikeCallable = Callable[[Unpack[tuple[torch.Tensor, ...]]], tuple[torch.Tensor, ...]]
OperatorLikeCallable = Callable[[torch.Tensor], tuple[torch.Tensor]]


@overload
def cg(
    operator: OperatorLikeCallable,
    right_hand_side: tuple[torch.Tensor] | torch.Tensor,
    *,
    initial_value: tuple[torch.Tensor] | torch.Tensor | None = None,
    preconditioner_inverse: OperatorLikeCallable | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor]: ...


@overload
def cg(
    operator: OperatorMatrixLikeCallable,
    right_hand_side: Sequence[torch.Tensor],
    *,
    initial_value: Sequence[torch.Tensor] | None = None,
    preconditioner_inverse: OperatorMatrixLikeCallable | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]: ...


def cg(
    operator: LinearOperator | LinearOperatorMatrix | OperatorMatrixLikeCallable | OperatorLikeCallable,
    right_hand_side: Sequence[torch.Tensor] | torch.Tensor,
    *,
    initial_value: Sequence[torch.Tensor] | torch.Tensor | None = None,
    preconditioner_inverse: LinearOperator
    | LinearOperatorMatrix
    | OperatorMatrixLikeCallable
    | OperatorLikeCallable
    | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...] | tuple[torch.Tensor]:
    r"""(Preconditioned) Conjugate Gradient for solving :math:`Hx=b`.

    This algorithm solves systems of the form :math:`H x = b`, where :math:`H` is a self-adjoint positive semidefinite
    linear operator and :math:`b` is the right-hand side.

     The method performs the following steps:

     1. Initialize the residual :math:`r_0 = b - Hx_0` (with :math:`x_0` as the initial guess).
     2. Set the search direction :math:`p_0 = r_0`.
     3. For each iteration :math:`k = 0, 1, 2, ...`:

        - Compute :math:`\alpha_k = \frac{r_k^T r_k}{p_k^T H p_k}`.
        - Update the solution: :math:`x_{k+1} = x_k + \alpha_k p_k`.
        - Compute the new residual: :math:`r_{k+1} = r_k - \alpha_k H p_k`.
        - Update the search direction: :math:`\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}`,
            then :math:`p_{k+1} = r_{k+1} + \beta_k p_k`.

    The operator can be either a `LinearOperator` or a `LinearOperatorMatrix`.
    In both cases, this implementation does not verify if the operator is self-adjoint and positive semidefinite.
    It will silently return the wrong result if the assumptions are not met.
    It can solve a batch of :math:`N` systems jointly if `right_hand_side` has a batch dimension
    and the operator interprets the batch dimension as :math:`H` being block-diagonal with blocks :math:`H_i`
    resulting in :math:`b = [b_1, ..., b_N] ^T`.

    If `preconditioner_inverse` is provided, it solves :math:`M^{-1}Hx = M^{-1}b`
    implicitly, where `preconditioner_inverse(r)` computes :math:`M^{-1}r`.

    See [Hestenes1952]_, [Nocedal2006]_, and [WikipediaCG]_ for more information.


    Parameters
    ----------
    operator
        Self-adjoint operator :math:`H`
    right_hand_side
        Right-hand-side :math:`b`.
    initial_value
        Initial guess :math:`x_0`. If `None`, the initial guess is set to the zero vector.
    preconditioner_inverse
        Preconditioner :math:`M^{-1}`. If None, no preconditioning is applied.
    max_iterations
        Maximum number of iterations.
    tolerance
        Tolerance for the L2 norm of the residual :math:`\|r_k\|_2`.
    callback
        Function called at each iteration. See `CGStatus`.

    Returns
    -------
        An approximate solution of the linear system :math:`Hx=b`.

    References
    ----------
    .. [Hestenes1952] Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems.
       Journal of Research of the National Bureau of Standards , 49(6), 409-436
    .. [Nocedal2006] Nocedal, J. (2006). *Numerical Optimization* (2nd ed.). Springer.
    .. [WikipediaCG] Wikipedia: Conjugate Gradient https://en.wikipedia.org/wiki/Conjugate_gradient
    """
    right_hand_side_ = (right_hand_side,) if isinstance(right_hand_side, torch.Tensor) else tuple(right_hand_side)
    if initial_value is None:  # start with zero initial value
        solution = tuple(torch.zeros_like(r) for r in right_hand_side_)
        residual = right_hand_side_
    else:  # start with provided initial value
        solution = (initial_value,) if isinstance(initial_value, torch.Tensor) else tuple(initial_value)
        if len(solution) != len(right_hand_side_):
            raise ValueError('Length mismatch in initial_value and right_hand_side')
        if any(s.shape != r.shape for s, r in zip(solution, right_hand_side_, strict=True)):
            raise ValueError('Shape mismatch in initial_value and right_hand_side')
        residual = tuple(rhs - op_sol for rhs, op_sol in zip(right_hand_side_, operator(*solution), strict=True))

    if preconditioner_inverse is not None:
        conjugate = preconditioner_inverse(*residual)
    else:
        conjugate = residual

    # dummy value. new value will be set in loop before first usage
    direction_dot_residual_old: torch.Tensor | None = None

    for iteration in range(max_iterations):
        residual_dot_residual = vdot(residual, residual).real

        if residual_dot_residual < tolerance**2:  # are we done?
            break

        if preconditioner_inverse is not None:
            direction = preconditioner_inverse(*residual)
            direction_dot_residual = vdot(residual, direction).real
        else:
            direction = residual
            direction_dot_residual = residual_dot_residual

        if direction_dot_residual_old is not None:  # not first iteration
            beta = direction_dot_residual / direction_dot_residual_old
            conjugate = tuple(d + beta * con for d, con in zip(direction, conjugate, strict=True))
        operator_conjugate = operator(*conjugate)
        alpha = direction_dot_residual / vdot(conjugate, operator_conjugate).real
        solution = tuple(sol + alpha * con for sol, con in zip(solution, conjugate, strict=True))
        residual = tuple(res - alpha * op_con for res, op_con in zip(residual, operator_conjugate, strict=True))
        direction_dot_residual_old = direction_dot_residual

        if callback is not None:
            continue_iterations = callback(
                {
                    'solution': solution,
                    'iteration_number': iteration,
                    'residual': residual,
                    'preconditioned_residual': direction,
                }
            )
            if continue_iterations is False:
                break

    return solution
