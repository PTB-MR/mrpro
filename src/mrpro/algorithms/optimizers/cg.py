"""Conjugate Gradient for linear systems with self-adjoint linear operator."""

from collections.abc import Callable, Sequence

import torch
from typing_extensions import TypeVarTuple, Unpack, overload

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.LinearOperatorMatrix import LinearOperatorMatrix

Tuple = TypeVarTuple('Tuple')


def vdot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    """Vector dot product."""
    return sum(
        (torch.vdot(a_i.flatten(), b_i.flatten()) for a_i, b_i in zip(a, b, strict=True)), start=torch.tensor(0.0)
    )


class CGStatus(OptimizerStatus):
    """Conjugate gradient callback base class."""

    residual: tuple[torch.Tensor, ...]
    """Residual of the current estimate."""


@overload
def cg(
    operator: LinearOperator,
    right_hand_side: torch.Tensor,
    *,
    initial_value: torch.Tensor | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> torch.Tensor: ...


@overload
def cg(
    operator: LinearOperatorMatrix | Callable[[Unpack[tuple[torch.Tensor, ...]]], tuple[torch.Tensor, ...]],
    right_hand_side: Sequence[torch.Tensor],
    *,
    initial_value: Sequence[torch.Tensor] | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]: ...


def cg(
    operator: LinearOperator
    | LinearOperatorMatrix
    | Callable[[Unpack[tuple[torch.Tensor, ...]]], tuple[torch.Tensor, ...]],
    right_hand_side: torch.Tensor | Sequence[torch.Tensor],
    *,
    initial_value: torch.Tensor | Sequence[torch.Tensor] | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    r"""CG for solving a linear system :math:`Hx=b`.

    This algorithm solves systems of the form :math:`H x = b`, where :math:`H` is a self-adjoint linear operator
    and :math:`b` is the right-hand side. The method can solve a batch of :math:`N` systems jointly, thereby taking
    :math:`H` as a block-diagonal with blocks :math:`H_i` and :math:`b = [b_1, ..., b_N] ^T`.

     The method performs the following steps:

     1. Initialize the residual :math:`r_0 = b - Hx_0` (with :math:`x_0` as the initial guess).
     2. Set the search direction :math:`p_0 = r_0`.
     3. For each iteration :math:`k = 0, 1, 2, ...`:

        - Compute :math:`\alpha_k = \frac{r_k^T r_k}{p_k^T H p_k}`.
        - Update the solution: :math:`x_{k+1} = x_k + \alpha_k p_k`.
        - Compute the new residual: :math:`r_{k+1} = r_k - \alpha_k H p_k`.
        - Update the search direction: :math:`\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}`,
            then :math:`p_{k+1} = r_{k+1} + \beta_k p_k`.

    This implementation assumes that :math:`H` is self-adjoint and does not verify this condition.

    See [Hestenes1952]_, [Nocedal2006]_, and [WikipediaCG]_ for more information.

    Parameters
    ----------
    operator
        self-adjoint operator (named :math:`H` above)
    right_hand_side
        right-hand-side of the system (named :math:`b` above)
    initial_value
        initial value of the algorithm; if `None`, it will be set to `right_hand_side`
    max_iterations
        maximal number of iterations. Can be used for early stopping.
    tolerance
        tolerance for the residual; if set to zero, the maximal number of iterations
        is the only stopping criterion used to stop the cg.
        If the condition number of :math:`H` is large, a small residual may not imply a highly accurate solution.
    callback
        Function to be called at each iteration. This can be used to monitor the progress of the algorithm.
        If it returns `False`, the algorithm stops at that iteration.

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
    if isinstance(operator, LinearOperator):
        operator_ = LinearOperatorMatrix.from_diagonal(operator)
    else:
        operator_ = operator

    if not isinstance(right_hand_side, torch.Tensor):
        right_hand_side_ = tuple(right_hand_side)
    else:
        right_hand_side_ = (right_hand_side,)

    if initial_value is None:
        solution: tuple[torch.Tensor, ...] = tuple(x.clone() for x in right_hand_side_)
    else:
        initial_value_ = (initial_value,) if isinstance(initial_value, torch.Tensor) else initial_value
        if len(initial_value_) != len(right_hand_side_):
            raise ValueError(
                'Length of initial_value and right_hand_side must match. '
                f'Got {len(initial_value_), len(right_hand_side_)}'
            )
        if any(i.shape != r.shape for i, r in zip(initial_value_, right_hand_side_, strict=True)):
            raise ValueError(
                'Shapes of initial_value and right_hand_side must match. '
                f'Got {[i.shape for i in initial_value_], [r.shape for r in right_hand_side_]}'
            )

        solution = tuple(i.clone() for i in initial_value_)

    residual = tuple(rhs - op_sol for rhs, op_sol in zip(right_hand_side_, operator_(*solution), strict=True))
    conjugate_vector = tuple(r.clone() for r in residual)

    # dummy value. new value will be set in loop before first usage
    residual_norm_squared_previous: torch.Tensor | None = None

    for iteration in range(max_iterations):
        residual_norm_squared = vdot(residual, residual).real

        if residual_norm_squared < tolerance**2:
            break

        if residual_norm_squared_previous is not None:  # not first iteration
            beta = residual_norm_squared / residual_norm_squared_previous
            conjugate_vector = tuple(r + beta * c for r, c in zip(residual, conjugate_vector, strict=True))

        operator_conjugate_vector = operator_(*conjugate_vector)
        alpha = residual_norm_squared / vdot(conjugate_vector, operator_conjugate_vector).real
        solution = tuple(s + alpha * c for s, c in zip(solution, conjugate_vector, strict=True))
        residual = tuple(r - alpha * op_c for r, op_c in zip(residual, operator_conjugate_vector, strict=True))
        residual_norm_squared_previous = residual_norm_squared

        if callback is not None:
            continue_iterations = callback(
                {
                    'solution': solution,
                    'iteration_number': iteration,
                    'residual': residual,
                }
            )
            if continue_iterations is False:
                break
    if (
        isinstance(operator, LinearOperator)
        and isinstance(right_hand_side, torch.Tensor)
        and (initial_value is None or isinstance(initial_value, torch.Tensor))
    ):
        # For backward compatibility if called with a single tensor and operator.
        return solution[0]
    return solution
