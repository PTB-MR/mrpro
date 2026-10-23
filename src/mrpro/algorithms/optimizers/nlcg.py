"""Nonlinear Conjugate Gradient."""

from collections.abc import Callable, Sequence

import torch


def nonlinear_cg(
    f: Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
    initial_parameters: Sequence[torch.Tensor],
    lr: float = 1.0,
    max_iter: int = 100,
    tolerance_grad: float = 1e-07,
    tolerance_change: float = 1e-09,
    callback: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Nonlinear Conjugate Gradient optimizer for non-linear minimization problems.

    Parameters
    ----------
    f
        Scalar function to be minimized.
    initial_parameters
        Sequence of parameters to be optimized. These parameters will not be modified directly;
        copies are made.
    lr
        Learning rate (used as an initial step size in the line search).
    max_iter
        Maximum number of iterations.
    tolerance_grad
        Termination tolerance on the gradient norm.
    tolerance_change
        Termination tolerance on the parameter change.
    callback
        Function to be called after each iteration (optional).

    Returns
    -------
    Tuple of optimized parameters.
    """
    params = [p.clone().detach().requires_grad_(True) for p in initial_parameters]
    f_value = f(tuple(params))
    f_value.backward()

    grads = [p.grad.clone() for p in params]
    directions = [-g for g in grads]

    for _ in range(max_iter):
        alpha = lr
        while True:
            new_params = [p + alpha * d for p, d in zip(params, directions, strict=False)]
            new_f_value = f(tuple(new_params))
            if new_f_value <= f_value + 1e-4 * alpha * sum(
                (g * d).sum() for g, d in zip(grads, directions, strict=False)
            ):
                break
            alpha *= 0.5

        params = [p + alpha * d for p, d in zip(params, directions, strict=False)]
        f_value = f(tuple(params))

        if torch.sqrt(sum(torch.sum(g**2) for g in grads)) < tolerance_grad:
            break

        for p in params:
            p.grad = None
        f_value.backward()
        new_grads = [p.grad.clone() for p in params]

        if (
            torch.sqrt(sum(torch.sum((new_g - g) ** 2) for new_g, g in zip(new_grads, grads, strict=False)))
            < tolerance_change
        ):
            break

        # PR formula for beta
        beta = sum((new_g - g).dot(new_g) for new_g, g in zip(new_grads, grads, strict=False)) / sum(
            g.dot(g) for g in grads
        )
        directions = [-new_g + beta * d for new_g, d in zip(new_grads, directions, strict=False)]
        grads = new_grads

        if callback:
            callback(tuple(params))

    return tuple(params)
