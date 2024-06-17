"""Proximal Gradient Descent algorithm."""
# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math

import torch

from mrpro.operators._Functional import Functional


def grad_and_value(function, x, create_graph=False):
    def inner(x):
        if create_graph and x.requires_grad:
            # shallow clone
            xg = x.view_as(x)

        xg = x.detach().requires_grad_(True)
        (y,) = function(xg)

        yg = y if isinstance(y, torch.Tensor) else y[0]
        grad = torch.autograd.grad(yg, xg)[0]
        return grad, y

    return inner(x)


def pgd(
    f: Functional,
    g: Functional,
    initial_value: torch.Tensor,
    stepsize: float = 1.0,
    reg_parameter: float = 0.01,
    max_iterations: int = 128,
    backtrack_factor: float = 1.0,
) -> torch.Tensor:
    """Proximal gradient descent algorithm for solving problem min_x f(x) + g(x).

    It relies on the implementation of the proximal map of g.

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
            gradient, f_y = grad_and_value(f, y)
            (x,) = g.prox(y - stepsize * gradient, reg_parameter * stepsize)

            if not backtracking:
                # no need to check stepsize, continue to next iteration
                break
            difference = x - y
            Q = (
                f_y
                + 1 / (2 * stepsize) * difference.abs().square().sum()
                + torch.vdot(gradient.flatten(), difference.flatten()).real
            )

            (f_x,) = f(x)
            if f_x <= Q:
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
        y = x + (t - 1.0) / t * (x - x_old)

        # update x and  t
        x_old = x
        t_old = t

    return y