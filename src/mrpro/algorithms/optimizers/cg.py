"""Conjugate Gradient for linear systems with self-adjoint linear operator."""

from collections.abc import Callable, Sequence

import torch
from typing_extensions import TypeVarTuple, Unpack, overload

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from ...operators.IdentityOp import IdentityOp
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

    preconditioned_residual: tuple[torch.Tensor, ...]
    """Preconditioned residual of the current estimate."""


TensorTuple = TypeVarTuple('TensorTuple')
OperatorMatrixLikeCallable = Callable[[Unpack[TensorTuple]], tuple[Unpack[TensorTuple]]]
OperatorLikeCallable = Callable[[torch.Tensor],tuple[torch.Tensor]]

@overload
def cg(
    operator: OperatorLikeCallable
    right_hand_side: tuple[torch.Tensor],
    *,
    initial_value: tuple[torch.Tensor] | None = None,
    preconditioner_inverse: OperatorLikeCallable| None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor]: ...


@overload
def cg(
    operator: Callable[[Unpack[tuple[torch.Tensor, ...]]], tuple[torch.Tensor, ...]],
    right_hand_side: tuple[Unpack[TensorTuple]]|Sequence[torch.Tensor],
    *,
    initial_value: tuple[Unpack[TensorTuple]]|Sequence[torch.Tensor]|None=None,
    preconditioner_inverse: OperatorLikeCallable|None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[Unpack[TensorTuple]]: ...


def cg(
    operator: LinearOperator
    | LinearOperatorMatrix
    | OperatorMatrixLikeCallable|OperatorLikeCallable,
    right_hand_side:Sequence[torch.Tensor],
    *,
    initial_value: Sequence[torch.Tensor] | None = None,
    preconditioner_inverse: LinearOperator
    | LinearOperatorMatrix
    | OperatorMatrixLikeCallable|OperatorLikeCallable|None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""(Preconditioned) Conjugate Gradient for solving :math:`Hx=b`.

    Solves systems where :math:`H` is self-adjoint (and ideally positive definite).
    If `preconditioner_inverse` is provided, it solves :math:`M^{-1}Hx = M^{-1}b`
    implicitly, where `preconditioner_inverse(r)` computes :math:`M^{-1}r`.

    Parameters
    ----------
    operator
        Self-adjoint operator :math:`H`.
    right_hand_side
        Right-hand-side :math:`b`.
    initial_value
        Initial guess :math:`x_0`.
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

    if initial_value is None:
        solution= right_hand_side
    elif len(initial_value) != len(right_hand_side):
            raise ValueError('Length mismatch in initial_value and right_hand_side')
    elif any(i.shape != r.shape for i, r in zip(initial_value, right_hand_side, strict=True)):
            raise ValueError('Shape mismatch in initial_value and right_hand_side')
    else:
        solution = initial_value

    residual = tuple(rhs - op_sol for rhs, op_sol in zip(right_hand_side, operator(*solution), strict=True))

    if preconditioner_inverse is not None:
        conjugate_vector = preconditioner_inverse(residual)
    else:
        conjugate_vector = residual



    # dummy value. new value will be set in loop before first usage
    residual_norm_squared_previous: torch.Tensor | None = None

    for iteration in range(max_iterations):
        residual_norm_squared = vdot(residual, residual).real

        if residual_norm_squared < tolerance**2:
            break

        if preconditioner_inverse is not None:
            preconditioned_residual = preconditioner_inverse(residual)
            preconditioned_dot_residual = vdot(residual, preconditioned_residual).real
        else:
            preconditioned_dot_residual = residual_norm_squared
            preconditioned_residual = residual


        if residual_norm_squared_previous is not None:  # not first iteration
            beta = residual_norm_squared / preconditioned_dot_residual
            conjugate_vector = tuple(r + beta * c for r, c in zip(preconditioned_residual, conjugate_vector, strict=True))

        operator_conjugate_vector = operator(*conjugate_vector)
        alpha = preconditioned_dot_residual / vdot(conjugate_vector, operator_conjugate_vector).real
        solution = tuple(s + alpha * c for s, c in zip(solution, conjugate_vector, strict=True))
        residual = tuple(r - alpha * op_c for r, op_c in zip(residual, operator_conjugate_vector, strict=True))
        residual_norm_squared_previous = residual_norm_squared

        if callback is not None:
            continue_iterations = callback(
                {
                    'solution': solution,
                    'iteration_number': iteration,
                    'residual': residual,
                    'preconditioned_residual': preconditioned_residual
                }
            )
            if continue_iterations is False:
                break

    return solution
