"""Stabilized Bi-Conjugate Gradient method for non-symmetric linear systems."""

from collections.abc import Callable, Sequence
from warnings import warn

import torch
from typing_extensions import TypedDict, Unpack

from mr2.operators import MultiIdentityOp
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.LinearOperatorMatrix import LinearOperatorMatrix


def vdot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], force_real: bool = False) -> torch.Tensor:
    """Vector dot product."""
    if force_real:
        a = tuple(torch.view_as_real(a_i) if a_i.is_complex() else a_i for a_i in a)
        b = tuple(torch.view_as_real(b_i) if b_i.is_complex() else b_i for b_i in b)
    return sum(
        (torch.vdot(a_i.flatten(), b_i.flatten()) for a_i, b_i in zip(a, b, strict=True)), start=torch.tensor(0.0)
    )


OperatorMatrixLikeCallable = Callable[[Unpack[tuple[torch.Tensor, ...]]], tuple[torch.Tensor, ...]]


class BiCGStatus(TypedDict):
    """BiCG callback status."""

    solution: tuple[torch.Tensor, ...]
    iteration_number: int
    residual: tuple[torch.Tensor, ...]


def bicg(
    operator: LinearOperator | LinearOperatorMatrix | OperatorMatrixLikeCallable,
    right_hand_side: Sequence[torch.Tensor],
    *,
    initial_value: Sequence[torch.Tensor] | None = None,
    preconditioner_inverse: LinearOperator | LinearOperatorMatrix | OperatorMatrixLikeCallable | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-6,
    callback: Callable[[BiCGStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""(Preconditioned) Bi-Conjugate Gradient Stabilized method for :math:`Hx=b`.

    This algorithm solves systems of the form :math:`H x = b`, where :math:`H` is a
    linear operator (potentially non-symmetric) and :math:`b` is the right-hand side.
    The method can solve multiple systems jointly if the operator and tensors handle batch dimensions.

    The method performs the following steps (simplified):

    1. Initialize :math:`x_0`, compute residual :math:`r_0 = b - Hx_0`.
       Set shadow residual :math:`\hat{r}_0 = r_0`.
    2. Initialize :math:`\rho_0 = \alpha = \omega = 1`, :math:`p_0 = v_0 = 0`.
    3. For iteration :math:`k = 0, 1, 2, ...`:
        1. Compute inner product: :math:`\rho_k = \hat{r}_0^H r_{k-1}`.
        2. Compute step direction modifier: :math:`\beta_{k-1} = (\rho_k / \rho_{k-1}) (\alpha_{k-1} / \omega_{k-1})`.
        3. Update search direction: :math:`p_k = r_{k-1} + \beta_{k-1} (p_{k-1} - \omega_{k-1} v_{k-1})`.
        4. Apply preconditioner: :math:`y_k = M^{-1} p_k`.
        5. Compute operator applied to search direction: :math:`v_k = H y_k`.
        6. Compute step size :math:`\alpha_k = \rho_k / (\hat{r}_0^H v_k)`.
        7. Compute intermediate residual: :math:`s_k = r_{k-1} - \alpha_k v_k`.
        8. Check convergence: If :math:`\|s_k\|_2` is small, update :math:`x_k = x_{k-1} + \alpha_k y_k` and stop.
        9. Apply preconditioner to intermediate residual: :math:`z_k = M^{-1} s_k`.
        10. Compute operator applied to preconditioned intermediate residual: :math:`t_k = H z_k`.
        11. Compute stabilization factor: :math:`\omega_k = (t_k^H s_k) / (t_k^H t_k)`.
        12. Update solution: :math:`x_k = x_{k-1} + \alpha_k y_k + \omega_k z_k`.
        13. Update residual: :math:`r_k = s_k - \omega_k t_k`.
        14. Update :math:`\rho_{k-1} = \rho_k` for the next iteration.

    If `preconditioner_inverse` is provided, it solves :math:`M^{-1}Hx = M^{-1}b`
    implicitly, where `preconditioner_inverse(r)` computes :math:`M^{-1}r`.

    See [VanDerVorst1992]_ and [WikipediaBiCGSTAB]_ for more details.

    Parameters
    ----------
    operator
        Linear operator :math:`H`. Does not need to be self-adjoint.
    right_hand_side
        Right-hand-side :math:`b`. Must be a sequence of tensors.
    initial_value
        Initial guess :math:`x_0`. If None, defaults to zeros. Must be a sequence of tensors.
    preconditioner_inverse
        Preconditioner :math:`M^{-1}`. If None, no preconditioning is applied (identity operator).
    max_iterations
        Maximum number of iterations.
    tolerance
        Tolerance for the L2 norm of the residual :math:`\|r_k\|_2`.
    callback
        Function called at the end of each iteration. See `BiCGSTABStatus`.
        If the callback returns `False`, the iteration stops.

    Returns
    -------
        An approximate solution :math:`x` of the linear system :math:`Hx=b` as a tuple of tensors.

    References
    ----------
    .. [VanDerVorst1992] Van der Vorst, H. A. (1992). Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the
       Solution of Nonsymmetric Linear Systems. SIAM Journal on Scientific and Statistical Computing, 13(2), 631-644.
    .. [WikipediaBiCGSTAB] Wikipedia: Bi-conjugate gradient stabilized method https://en.wikipedia.org/wiki/Bi-conjugate_gradient_stabilized_method
    """
    right_hand_side_ = tuple(right_hand_side)
    if tolerance < torch.finfo(right_hand_side[0].dtype).eps:
        warn(f'Tolerance is very small ({tolerance}), this can cause nan values.', stacklevel=2)
    if not right_hand_side_:
        raise ValueError('right_hand_side cannot be empty')

    if initial_value is None:
        solution = tuple(torch.zeros_like(r) for r in right_hand_side_)
    else:
        solution = tuple(initial_value)
        if len(solution) != len(right_hand_side_):
            raise ValueError('Length mismatch in initial_value and right_hand_side')
        if any(s.shape != r.shape for s, r in zip(solution, right_hand_side_, strict=True)):
            raise ValueError('Shape mismatch in initial_value and right_hand_side')
    if any(r.is_complex() for r in right_hand_side_) and not all(r.is_complex() for r in right_hand_side_):
        force_real = True
    else:
        force_real = False
    residual = tuple(r - op_sol for r, op_sol in zip(right_hand_side_, operator(*solution), strict=True))
    arbitrary = tuple(r for r in residual)

    preconditioner = preconditioner_inverse if preconditioner_inverse is not None else MultiIdentityOp()
    alpha = omega = previous_product = operator_search = None
    search_direction = residual

    for iteration in range(max_iterations):
        if vdot(residual, residual, force_real).real < tolerance**2:
            break

        arbitrary_dot_residual = vdot(arbitrary, residual, force_real)
        if (
            previous_product is not None and alpha is not None and omega is not None and operator_search is not None
        ):  # not first iteration
            beta = (arbitrary_dot_residual / previous_product) * (alpha / omega)
            search_direction = tuple(
                r + beta * (p - omega * v) for r, p, v in zip(residual, search_direction, operator_search, strict=True)
            )
        preconditioned_search = preconditioner(*search_direction)
        operator_search = operator(*preconditioned_search)
        alpha = arbitrary_dot_residual / vdot(arbitrary, operator_search, force_real)

        interim = tuple(r - alpha * v for r, v in zip(residual, operator_search, strict=True))
        preconditioned_interim = preconditioner(*interim)
        operator_interim = operator(*preconditioned_interim)
        omega = vdot(operator_interim, interim, force_real) / vdot(operator_interim, operator_interim, force_real).real

        solution = tuple(
            x + alpha * p + omega * s
            for x, p, s in zip(solution, preconditioned_search, preconditioned_interim, strict=True)
        )

        residual = tuple(i - omega * t for i, t in zip(interim, operator_interim, strict=True))
        previous_product = arbitrary_dot_residual

        if callback is not None:
            continue_iterations = callback({'solution': solution, 'iteration_number': iteration, 'residual': residual})
            if continue_iterations is False:
                break

    return solution
