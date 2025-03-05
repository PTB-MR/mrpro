"""Proximal Gradient Descent algorithm."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from typing_extensions import Unpack

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrpro.operators import ProximableFunctionalSeparableSum
from mrpro.operators.Functional import ProximableFunctional
from mrpro.operators.Operator import Operator


@dataclass
class PGDStatus(OptimizerStatus):
    """Proximal Gradient Descent callback base class."""

    stepsize: float | torch.Tensor
    objective: Callable[..., torch.Tensor]


def pgd(
    f: Operator[torch.Tensor, tuple[torch.Tensor]] | Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor]],
    g: ProximableFunctional | ProximableFunctionalSeparableSum,
    initial_value: torch.Tensor | tuple[torch.Tensor, ...],
    stepsize: float = 1.0,
    max_iterations: int = 128,
    backtrack_factor: float = 1.0,
    convergent_iterates_variant: bool = False,
    callback: Callable[[PGDStatus], None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Proximal gradient descent algorithm for solving problem :math:`min_x f(x) + g(x)`.

    f is convex, differentiable, and with L-Lispchitz gradient.
    g is convex, possibly non-smooth with computable proximal map.

    For fixed stepsize t, pgd converges globally when :math:`t \in (0, 1/L)`,
    where L is the Lipschitz constant of the gradient of f.
    In applications, f is often of the form :math:`f(x) = 1/2 \|Ax - y\|^2`,
    where :math:`A` is a linear operator.
    In this case, :math:`t \in (0, 1/\|A\|_2^2)` for convergence.
    If no backtracking is used, the fixed stepsize should be given accordingly to the convergence condition.

    Example:
        L1 regularized image reconstruction. Problem formulation: :math:`min_x 1/2 ||Fx - y||^2 + \lambda ||x||_1`,
        with :math:`F` being the Fourier Transform, target denoting the acquired data \in k-space and x \in image space,
        :math:`f(x) = 1/2 \|Fx - y\|^2, g(x) = \lambda \|x\|_1`.
        In this case, :math:`||F^T F|| = 1`. ::
                kspace_data = torch.randn(3, 10, 10 , dtype=torch.complex64)
                fft = FastFourierOp()
                l2 = L2NormSquared(target=kspace_data)
                f = l2 @ fft
                reg_parameter = 0.01
                g = reg_parameter * L1Norm()
                operator_norm = 1.
                stepsize = 0.85 * 1 / operator_norm**2
                initial_value = torch.ones(3,10,10)
                pgd_image_solution = pgd(
                    f=f,
                    g=g,
                    initial_value=initial_value,
                    stepsize=stepsize,
                    max_iterations=200,
                    backtrack_factor=1.0,
                )

    References
    ----------
    .. [BE2009] Beck A, Teboulle M (2009) A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
       SIAM Journal on Imaging Sciences
       https://www.ceremade.dauphine.fr/~carlier/FISTA
    .. [CHAM2015] Chambolle A, Dossal C (2015) On the convergence of the iterates of "FISTA".
       Journal of Optimization Theory and Applications
       https://inria.hal.science/hal-01060130v3

    Parameters
    ----------
    f
        convex, differentiable functional
    g
        convex, possibly non-smooth functional with computable proximal map
    initial_value
        initial value for the solution of the algorithm
    stepsize
        stepsize needed in the gradient step, constant if no backtracking is used, otherwise it is the initial stepsize
    max_iterations
        number of iterations
    backtrack_factor
        must be :math:`<=1`. if :math:`<1.`, the backtracking rule for stepsize introduced by [BE2009]_ is used
    convergent_iterates_variant
        by default, the algorithm updates the variable t as originally described in [BE2009]_.
        If set to `True`, the algorithm updates t as suggested by [CHAM2015]_,
        i.e. at iteration :math:`n`, :math:`t_n = \frac{n+a-1}{a}`, with chosen :math:`a=3`.
        This choice ensures the theoretical convergence of solution.
    callback
        function to be called at each iteration

    Returns
    -------
        an approximate solution of the minimization problem
    """
    # check that backtracking factor is in the correct range
    if not 0.0 <= backtrack_factor <= 1.0:
        raise ValueError('Backtracking factor must be in the range [0, 1].')

    backtracking = not math.isclose(backtrack_factor, 1)

    if isinstance(initial_value, torch.Tensor):
        initial_values: tuple[torch.Tensor, ...] = (initial_value,)
    else:
        initial_values = initial_value

    if isinstance(g, ProximableFunctional):
        g_sum = ProximableFunctionalSeparableSum(g)
    else:
        g_sum = g

    x_old = initial_values
    y: tuple[torch.Tensor, ...] = initial_values
    gradient: tuple[torch.Tensor, ...]
    f_y: torch.Tensor
    t_old = 1.0
    grad_and_value_f = torch.func.grad_and_value(
        lambda x: f(*x)[0],
    )
    for iteration in range(max_iterations):
        while stepsize > 1e-30:
            gradient, f_y = grad_and_value_f(y)
            x = g_sum.prox(*[yi - stepsize * gi for yi, gi in zip(y, gradient, strict=True)], sigma=stepsize)

            if not backtracking:
                # no need to check stepsize, continue to next iteration
                break
            difference = tuple(xi - yi for xi, yi in zip(x, y, strict=True))
            quadratic_approx = (
                f_y
                + 1 / (2 * stepsize) * sum(di.abs().square().sum() for di in difference)
                + sum(torch.vdot(gi.flatten(), di.flatten()).real for gi, di in zip(gradient, difference, strict=True))
            )

            (f_x,) = f(*x)

            if f_x <= quadratic_approx:
                # stepsize is ok, continue to next iteration
                break
            stepsize *= backtrack_factor

        else:
            if backtracking:
                raise RuntimeError('After backtracking, the stepsize became to small.')
            else:
                raise RuntimeError('Stepsize to small.')

        if convergent_iterates_variant:
            t = (iteration + 2) / 3
        else:
            t = (1 + math.sqrt(1 + 4 * t_old**2)) / 2

        y = tuple(xi + (t_old - 1.0) / t * (xi - xi_old) for xi, xi_old in zip(x, x_old, strict=True))

        x_old = x
        t_old = t

        if callback is not None:
            callback(
                PGDStatus(
                    solution=x, iteration_number=iteration, stepsize=stepsize, objective=lambda *x: f(*x)[0] + g(*x)[0]
                )
            )

    return x
