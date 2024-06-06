"""FISTA algorithm."""
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


def grad(f: Functional, x: torch.Tensor):
    """Compute the gradient of f(x).

    Parameters
    ----------
    f
        differentiable function
    x
        point in the domain of f
    """
    x = x.clone().detach().requires_grad_(True)
    y = f(x)
    return torch.autograd.grad(y, x)[0]


def fista(
    f: Functional,
    g: Functional,
    initial_value: torch.Tensor,
    stepsize: float = 1.0,
    reg_parameter: float = 0.01,
    max_iterations: int = 128,
    backtrack_factor: float = 1.0,
) -> torch.Tensor:
    """FISTA algorithm for solving problem min_x f(x) + g(x).

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
    complex_input = torch.is_complex(initial_value)
    x_old = initial_value
    y = initial_value
    t_old = 1.0

    for _ in range(max_iterations):
        while stepsize > 1e-30:
            f_y = f(y)
            # calculate the proximal gradient step
            gradient = grad(f_y, y)
            x = g.prox(y - stepsize * gradient, reg_parameter * stepsize)

            if not backtracking:
                # no need to check stepsize, continue to next iteration
                break
            difference = x - y
            Q = f_y + 1 / (2 * stepsize) * difference.abs().square().sum()
            if complex_input:
                Q += torch.vdot(gradient.flatten(), difference.flatten()).real
            else:
                Q += torch.vdot(gradient.flatten(), difference.flatten())

            if f(x) <= Q:
                # stepsize is ok, continue to next iteration
                break
            stepsize *= backtrack_factor

        else:
            if backtracking:
                raise RuntimeError('After backtracking, the stepsize became to small.')
            else:
                raise RuntimeError('Stepsize to small.')

        # update fista timestep t
        t = (1 + math.sqrt(1 + 4 * t_old**2)) / 2

        # update the solution
        y = x + (t - 1.0) / t * (x - x_old)

        # update x and  t
        x_old = x
        t_old = t

    return y


##  ideas for examples
# example deblurring ?

# from mrpro.operators._LinearOperator import LinearOperator
# import torch.nn as nn
# import torch.nn.functional as F
# class BlurOperator(LinearOperator):
#     """Blur Operator."""
#     def __init__(
#         self,
#         kernel: torch.Tensor
#     ) -> None:
#         super().__init__()
#         self.kernel = kernel

#     def forward(self,x: torch.Tensor, padding=4) -> torch.Tensor:
#         return F.conv2d(x, self.kernel, padding)

#     def adjoint(self, y, padding=4):
#         return F.conv_transpose2d(y,self.kernel,padding=4)


# def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
#     """Generates a Gaussian kernel."""
#     x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
#     y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
#     x, y = torch.meshgrid(x, y)
#     kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
#     kernel /= kernel.sum()
#     return kernel.view(1, 1, size, size)

# kernel = gaussian_kernel(9, 2)
# B = BlurOperator(kernel=kernel)

# x = torch.randn(1,1,160,160)
# blurred_x = B(x)

# l2_dc = L2DataDiscrepancy(blurred_x, factor=0.5)
# f = l2_dc @ B
# g = L1Norm(...)

# s_initial = torch.zeros(1,1,160,160)
# stepsize = 0.5 * OperatorNorm(B.adjoint @ B)

# s_solution = fista(f=f,
#                  g=g,
#                 initial_value=s_initial,
#                 stepsize=stepsize,
#                 reg_parameter=0.01)

# s_solution = fista_with_backtracking(f=f,
#                  g=g,
#                 initial_value=s_initial,
#                 stepsize_initial=0.5,
#                 reg_parameter=0.01)

# x_solution = W.inverse(s_solution)


# example wavelet

# W = WaveletOperator(...)
# F = FourierOperator(...)
# A = F @ W.inverse

# y = ... # k-space data
# l2_dc = L2DataDiscrepancy(y, factor=0.5)
# f = l2_dc @ A
# g = L1Norm(...)

# s_initial = torch.zeros(...)
# stepsize = 0.5 * OperatorNorm(A @ A.adjoint)

# s_solution = fista(f=f,
#                  g=g,
#                 initial_value=s_initial,
#                 stepsize=stepsize,
#                 reg_parameter=0.01)

# s_solution = fista_with_backtracking(f=f,
#                  g=g,
#                 initial_value=s_initial,
#                 stepsize_initial=0.5,
#                 reg_parameter=0.01)

# x_solution = W.inverse(s_solution)
