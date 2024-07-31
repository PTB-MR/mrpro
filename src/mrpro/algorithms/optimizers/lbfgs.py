"""LBFGS for solving non-linear minimization problems."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch.optim import LBFGS

from mrpro.operators.Operator import Operator


def lbfgs(
    f: Operator[*tuple[torch.Tensor, ...], tuple[torch.Tensor]],
    initial_parameters: Sequence[torch.Tensor],
    lr: float = 1.0,
    max_iter: int = 100,
    max_eval: int | None = 100,
    tolerance_grad: float = 1e-07,
    tolerance_change: float = 1e-09,
    history_size: int = 10,
    line_search_fn: None | Literal['strong_wolfe'] = 'strong_wolfe',
) -> tuple[torch.Tensor, ...]:
    """LBFGS for non-linear minimization problems.

    Parameters
    ----------
    f
        scalar function to be minimized
    initial_parameters
        Sequence (for example list) of parameters to be optimized.
        Note that these parameters will not be changed. Instead, we create a copy and
        leave the initial values untouched.
    lr
        learning rate
    max_iter
        maximal number of iterations
    max_eval
        maximal number of evaluations of f per optimization step
    tolerance_grad
        termination tolerance on first order optimality
    tolerance_change
        termination tolerance on function value/parameter changes
    history_size
        update history size
    line_search_fn
        line search algorithm, either 'strong_wolfe' or None (meaning constant step size)

    Returns
    -------
        list of optimized parameters
    """
    parameters = [p.detach().clone().requires_grad_(True) for p in initial_parameters]
    optim = LBFGS(
        params=parameters,
        lr=lr,
        history_size=history_size,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn,
    )

    def closure():
        optim.zero_grad()
        (objective,) = f(*parameters)
        objective.backward()
        return objective

    # run lbfgs
    optim.step(closure)

    return tuple(parameters)
