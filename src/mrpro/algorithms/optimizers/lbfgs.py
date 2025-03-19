"""LBFGS for solving non-linear minimization problems."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch.optim import LBFGS

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrpro.operators.Operator import OperatorType


def lbfgs(
    f: OperatorType,
    initial_parameters: Sequence[torch.Tensor],
    learning_rate: float = 1.0,
    max_iterations: int = 100,
    max_evaluations: int | None = 100,
    tolerance_grad: float = 1e-07,
    tolerance_change: float = 1e-09,
    history_size: int = 10,
    line_search_fn: None | Literal['strong_wolfe'] = 'strong_wolfe',
    callback: Callable[[OptimizerStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""
    LBFGS for (non-linear) minimization problems.

    The Limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) algorithm is a quasi-Newton optimization method
    that approximates the inverse Hessian matrix using a limited memory of past gradients and updates.
    It is well-suited for high-dimensional problems and leverages curvature information
    for faster convergence compared to first-order methods such as `mrpro.algorithms.optimizers.adam`.

    The parameter update rule is:

    .. math::

        \theta_{k+1} = \theta_k - \alpha_k H_k \nabla f(\theta_k),

    where :math:`H_k` is a limited-memory approximation of the inverse Hessian,
    and :math:`\alpha_k` is the step size determined via line search (e.g., strong Wolfe conditions).

    The algorithm performs the following steps:

    1. Compute the gradient of the objective function.
    2. Approximate the inverse Hessian matrix :math:`H_k` using stored gradients and updates.
    3. Perform a line search to compute the step size :math:`\alpha_k`.
    4. Update the parameters.
    5. Store the latest gradient and update information.

    This implementation wraps PyTorch's `torch.optim.LBFGS` class.
    For more information, see [WIKI]_, [NOC1980]_, and [LIU1989]_.

    References
    ----------
    .. [NOC1980] Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
       *Mathematics of Computation*, 35(151), 773-782. https://doi.org/10.1090/S0025-5718-1980-0572855-7
    .. [LIU1989] Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS method for large scale optimization."
       *Mathematical Programming*, 45(1-3), 503-528. https://doi.org/10.1007/BF01589116
    .. [WIKI] Wikipedia: Limited-memory_BFGS https://en.wikipedia.org/wiki/Limited-memory_BFGS


    Parameters
    ----------
    f
        scalar function to be minimized
    initial_parameters
        `Sequence` of parameters to be optimized.
        Note that these parameters will not be changed. Instead, we create a copy and
        leave the initial values untouched.
    learning_rate
        learning rate. This should usually be left as ``1.0`` if a line search is used.
    max_iterations
        maximal number of iterations
    max_evaluations
        maximal number of evaluations of `f` per optimization step
    tolerance_grad
        termination tolerance on first order optimality
    tolerance_change
        termination tolerance on function value/parameter changes
    history_size
        update history size
    line_search_fn
        line search algorithm, either ``strong_wolfe`` or `None` (meaning constant step size)
    callback
        Function to be called after each iteration. This can be used to monitor the progress of the algorithm.
        If it returns `False`, the algorithm stops at that iteration.
        N.B. the callback is not called within the line search of LBFGS.
        You can use the information from the `~mrpro.algorithms.optimizers.OptimizerStatus`
        to display a progress bar.

    Returns
    -------
        List of optimized parameters.
    """
    parameters = tuple(p.detach().clone().requires_grad_(True) for p in initial_parameters)
    optim = LBFGS(
        params=parameters,
        lr=learning_rate,
        history_size=history_size,
        max_iter=max_iterations,
        max_eval=max_evaluations,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn,
    )

    iteration = 0

    def closure():
        nonlocal iteration
        optim.zero_grad()
        (objective,) = f(*parameters)
        objective.backward()
        return objective

    iteration = 0
    # run lbfgs
    while iteration < max_iterations:
        optim.step(closure)
        iteration += 1

        if callback is not None:
            continue_iterations = callback({'solution': parameters, 'iteration_number': iteration})
            if continue_iterations is False:
                break

    return parameters
