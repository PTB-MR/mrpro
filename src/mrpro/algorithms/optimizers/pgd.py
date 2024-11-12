"""Proximal Gradient Descent algorithm."""

import math

import torch

from mrpro.operators.Functional import ProximableFunctional
from mrpro.operators.Operator import Operator


# TODO: make it work with g:ProximableFunctionalSeparableSum and f:Operator with multiple inputs
def pgd(
    f: Operator[torch.Tensor, tuple[torch.Tensor]],
    g: ProximableFunctional,
    initial_value: torch.Tensor,
    stepsize: float = 1.0,
    reg_parameter: float = 0.01,
    max_iterations: int = 128,
    backtrack_factor: float = 1.0,
) -> torch.Tensor:
    r"""Proximal gradient descent algorithm for solving problem min_x f(x) + g(x).

    f is convex, differentiable, and with L-Lispchitz gradient.
    g is convex, non-smooth with computable proximal map.

    For fixed stepsize t, pgd converges globally when t \in (0, 1/L),
    where L is the Lipschitz constant of the gradient of f.
    In applications, f is usually of the form f(x) = 1/2 ||Ax - target||^2, where A is a linear operator.
    In this case, t \in (0, 1/||A^T A||) for convergence.
    If no backtracking is used, the fixed stepsize should be given accordingly to the convergence condition.

    Example:
        L1 regularized image reconstruction. Problem formulation: min_x 1/2 ||Fx - target||^2 + ||x||_1,
        with F being the Fourier Transform, target the acquired data \in k-space and x \in image space,
        f(x) = 1/2 ||Fx - target||^2, g(x) = ||x||_1.
        In this case, ||F^T F|| = 1. ::
                fft = FastFourierOp()
                l2 = L2NormSquared(target=kspace_data)
                f = l2 @ fft
                g = L1Norm()
                fft_norm = 1.
                stepsize = 0.85 * 1 / fft_norm
                initial_value = torch.ones(image_space_shape)
                pgd_image_solution = pgd(
                    f=f,
                    g=g,
                    initial_value=initial_value,
                    stepsize=stepsize,
                    reg_parameter=0.01,
                    max_iterations=200,
                    backtrack_factor=1.0,
                )


    Parameters
    ----------
    f
        convex, differentiable functional
    g
        convex, non-smooth functional with computable proximal map
    initial_value
        initial value for the solution of the algorithm
    stepsize
        stepsize needed in the gradient step, is constant throughout all
        iterations
    reg_parameter
        regularization parameter that multiplies g
    max_iterations, optional
        number of iterations
    backtrack_factor
        must be <=1. if <1., Backtracking rule for stepsize following https://www.ceremade.dauphine.fr/~carlier/FISTA
        is used

    Returns
    -------
        an approximate solution of the minimization problem
    """
    backtracking = not math.isclose(backtrack_factor, 1)
    x_old = initial_value
    y = initial_value
    t_old = 1.0

    for _ in range(max_iterations):
        while stepsize > 1e-30:
            # calculate the proximal gradient step
            gradient, f_y = torch.func.grad_and_value(f, y)
            (x,) = g.prox(y - stepsize * gradient, reg_parameter * stepsize)

            if not backtracking:
                # no need to check stepsize, continue to next iteration
                break
            difference = x - y
            quadratic_approx = (
                f_y
                + 1 / (2 * stepsize) * difference.abs().square().sum()
                + torch.vdot(gradient.flatten(), difference.flatten()).real
            )

            (f_x,) = f(x)
            if f_x <= quadratic_approx:
                # stepsize is ok, continue to next iteration
                break
            stepsize *= backtrack_factor

        else:
            if backtracking:
                raise RuntimeError('After backtracking, the stepsize became to small.')
            else:
                raise RuntimeError('Stepsize to small.')

        # update timestep t
        t = (1 + math.sqrt(1 + 4 * t_old**2)) / 2

        # update the solution
        y = x + (t_old - 1.0) / t * (x - x_old)

        # update x and  t
        x_old = x
        t_old = t

    return y
