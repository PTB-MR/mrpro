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

import torch
import math

from mrpro.operators._LinearOperator import LinearOperator


def fista(
    f: Functional,
    g: Functional,
    initial_value: torch.Tensor,
    stepsize: float = 1., 
    reg_parameter: float = 0.01,
    max_iterations: int = 128,
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
    max_iterations, optional
        number of iterations
    
    Returns
    -------
        an approximate solution of the minimization problem
    """
    x = initial_value
    y = x
    t = 1.
    
    for iteration in range(max_iterations):
        # calculate the proximal step
        x_update = g.prox(y - stepsize * torch.autograd.grad(f(y), y), 
                          reg_parameter) 
        
        # update the stepsize
        t_update = (1 + math.sqrt(1 + 4*t**2))/2
        
        # update the solution
        y = x_update + (t - 1.)/t_update * (x_update - x)
        
        # update x and stepsize t
        x = x_update
        t = t_update

    return y


def backtracking_stepsize_rule(f,g,stepsize_old, y, reg_parameter, eta=1.2):
    """Backtracking rule for stepsize following https://www.ceremade.dauphine.fr/~carlier/FISTA."""
    stepsize_new = stepsize_old
    prox_y = g.prox(y - stepsize_new * torch.autograd.grad(f(y), y), 
                    reg_parameter)
    difference = prox_y - y
    Q = torch.vdot(difference.flatten(),torch.autograd.grad(f(y),y).flatten()) + 0.5 * stepsize_new * torch.norm(difference, 2)**2
    f_prox_y = f(prox_y) 
    
    i = 0
    while f_prox_y - f(y) > Q:
        i += 1
        stepsize_new= eta**i * stepsize_old
        
        prox_y = g.prox(y - stepsize_new * torch.autograd.grad(f(y), y), reg_parameter)
        difference = prox_y - y
        Q = torch.vdot(difference.flatten(),torch.autograd.grad(f(y),y).flatten()) + 0.5 * stepsize_new * torch.norm(difference, 2)**2
        f_prox_y = f(prox_y) 
    
    return stepsize_new


def fista_with_backtracking(
    f: Functional,
    g: Functional,
    initial_value: torch.Tensor, 
    stepsize_initial: float = 1., 
    reg_parameter: float = 0.01,
    max_iterations: int = 128,
) -> torch.Tensor:
    """FISTA algorithm for solving problem min_x f(x) + g(x).

    This relies on the implementation of the proximal map of g.
    
    stepsize is updated in each iteration.
    
    Parameters
    ----------
    f
        convex, differentiable functional
    g
        convex, non-smooth functional with computable proximal map
    initial_value
        initial value for the solution of the algorithm
    stepsize_initial
        initial stepsize for gradient step
    max_iterations, optional
        maximal number of iterations
    
    Returns
    -------
        an approximate solution of the minimization problem
    """
    x = initial_value
    y = x
    t = 1.
    stepsize_old = stepsize_initial
    
    for iteration in range(max_iterations):
        # updte the stepsize 
        stepsize_prox = backtracking_stepsize_rule(f,g,stepsize_old, y, 
                                                   reg_parameter)
        
        # calculate the proximal step
        x_update = g.prox(y - stepsize_prox * torch.autograd.grad(f(y), y), 
                          reg_parameter)
        
        # update the stepsize
        t_update = (1 + math.sqrt(1 + 4*t**2))/2
        
        # update the solution
        y = x_update + (t - 1.)/t_update * (x_update - x)
        
        # update x and stepsize t
        x = x_update
        t = t_update
        stepsize_old = stepsize_prox

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

