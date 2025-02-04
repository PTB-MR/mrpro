"""Conjugate Gradient for linear systems with self-adjoint linear operator."""

from collections.abc import Callable

import torch

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrpro.operators.LinearOperator import LinearOperator


class CGStatus(OptimizerStatus):
    """Conjugate gradient callback base class."""

    residual: torch.Tensor
    """Residual of the current estimate."""


def cg(
    operator: LinearOperator,
    right_hand_side: torch.Tensor,
    initial_value: torch.Tensor | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable[[CGStatus], None] | None = None,
) -> torch.Tensor:
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
    if initial_value is not None and (initial_value.shape != right_hand_side.shape):
        raise ValueError(
            f'Shapes of starting_value and right_hand_side must match,got {initial_value.shape, right_hand_side.shape}'
        )

    # initial residual
    residual = right_hand_side - operator(initial_value)[0] if initial_value is not None else right_hand_side.clone()

    # initialize conjugate vector
    conjugate_vector = residual.clone()

    # assign starting value to the solution
    solution = initial_value.clone() if initial_value is not None else right_hand_side.clone()

    # for the case where the residual is exactly zero
    if torch.vdot(residual.flatten(), residual.flatten()) == 0:
        return solution

    # dummy value. new value will be set in loop before first usage
    residual_norm_squared_previous: torch.Tensor | None = None

    for iteration in range(max_iterations):
        # calculate the square norm of the residual
        residual_flat = residual.flatten()
        residual_norm_squared = torch.vdot(residual_flat, residual_flat).real

        # check if the solution is already accurate enough
        if tolerance != 0 and (residual_norm_squared < tolerance**2):
            return solution

        if residual_norm_squared_previous is not None:  # not first iteration
            beta = residual_norm_squared / residual_norm_squared_previous
            conjugate_vector = residual + beta * conjugate_vector

        # update estimates of the solution and the residual
        (operator_conjugate_vector,) = operator(conjugate_vector)
        alpha = residual_norm_squared / (torch.vdot(conjugate_vector.flatten(), operator_conjugate_vector.flatten()))
        solution = solution + alpha * conjugate_vector
        residual = residual - alpha * operator_conjugate_vector

        residual_norm_squared_previous = residual_norm_squared

        if callback is not None:
            callback(
                {
                    'solution': (solution,),
                    'iteration_number': iteration,
                    'residual': residual,
                }
            )

    return solution
