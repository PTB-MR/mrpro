"""Tests for the conjugate gradient method."""

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

#%%
import pytest
import torch


# from tests import RandomGenerator 
from mrpro.phantoms import EllipsePhantom
from mrpro.operators import FastFourierOp
from mrpro.data._SpatialDimension import SpatialDimension
from mrpro.algorithms.optimizers import pgd


from mrpro.operators.functionals import MSEDataDiscrepancy
from mrpro.operators.functionals import L1Norm
#%%

def test_pgd_convergence_fft_example():
    dim = SpatialDimension.from_array_xyz((100,100,0))  
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)
    
    mse_discr = MSEDataDiscrepancy(data=kspace)
    f = mse_discr @ fft
    g = L1Norm()
    
    initial_value = torch.ones_like(image)
    pgd_solution = pgd(f=f, g=g, initial_value=initial_value,
                        stepsize=.1, reg_parameter= 0.01,
                        max_iterations=100,
                        backtrack_factor=1.0)
  
    #torch.testing.assert_close(pgd_solution, image, rtol=5e-3, atol=5e-3)
    assert f(pgd_solution)[0] + g(pgd_solution)[0] < f(initial_value)[0] + g(initial_value)[0]

def test_pgd_convergence_backtracking_fft_example():
    dim = SpatialDimension.from_array_xyz((100,100,0))  
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)
    
    mse_discr = MSEDataDiscrepancy(data=kspace)
    f = mse_discr @ fft
    g = L1Norm()
    
    initial_value = torch.ones_like(image)
    pgd_solution = pgd(f=f, g=g, initial_value=initial_value,
                        stepsize=1.0, reg_parameter= 0.01,
                        max_iterations=100, 
                        backtrack_factor=.75)
  
    #torch.testing.assert_close(pgd_solution, image, rtol=5e-3, atol=5e-3)
    assert f(pgd_solution)[0] + g(pgd_solution)[0] < f(initial_value)[0] + g(initial_value)[0]

# %%

def test_pgd_convergence_backtracking_denoising_example():
    dim = SpatialDimension.from_array_xyz((100,100,0))  
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    noise = torch.randn_like(image)
    noisy_image = image + noise
    
    mse_discr = MSEDataDiscrepancy(data=noisy_image)
    f = mse_discr
    g = L1Norm()
    
    initial_value = torch.ones_like(image)
    pgd_solution = pgd(f=f, g=g, initial_value=initial_value,
                        stepsize=1.0, reg_parameter= 0.01,
                        max_iterations=100, backtrack_factor=.75)

    assert f(pgd_solution)[0] + g(pgd_solution)[0] < f(initial_value)[0] + g(initial_value)[0]

#%%
# visualize
# import matplotlib.pyplot as plt
# import math
# def grad_and_value(function, x, create_graph=False):
#     def inner(x):
#         if create_graph and x.requires_grad:
#             # shallow clone
#             xg = x.view_as(x)

#         xg = x.detach().requires_grad_(True)
#         (y,) = function(xg)

#         yg = y if isinstance(y, torch.Tensor) else y[0]
#         grad = torch.autograd.grad(yg, xg)[0]
#         return grad, y

#     return inner(x)


# dim = SpatialDimension.from_array_xyz((100,100,0))  
# ellipse = EllipsePhantom()
# image = ellipse.image_space(dim)
# fft = FastFourierOp()
# (kspace,) = fft(image)
    
# mse_discr = MSEDataDiscrepancy(data=kspace)
# f = mse_discr @ fft
# g = L1Norm()
    
# initial_value = torch.ones_like(image)

# pgd_solution = pgd(f=f, g=g, initial_value=initial_value,
#                         stepsize=.5, reg_parameter= 0.01,
#                         max_iterations=10,
#                         backtrack_factor=.75)

# plt.imshow(pgd_solution[0,0,0].abs())
# plt.colorbar()
# (f_sol,) = f(pgd_solution)
# (g_sol,) = g(pgd_solution)
# print('value = ', f_sol + g_sol)

# pgd_solution = pgd(f=f, g=g, initial_value=initial_value,
#                         stepsize=.5, reg_parameter= 0.01,
#                         max_iterations=100,
#                         backtrack_factor=.75)

# plt.imshow(pgd_solution[0,0,0].abs())
# plt.colorbar()
# (f_sol,) = f(pgd_solution)
# (g_sol,) = g(pgd_solution)
# print('value = ', f_sol + g_sol)


# pgd_solution = pgd(f=f, g=g, initial_value=initial_value,
#                         stepsize=.5, reg_parameter= 0.01,
#                         max_iterations=150,
#                         backtrack_factor=.75)

# plt.imshow(pgd_solution[0,0,0].abs())
# plt.colorbar()
# (f_sol,) = f(pgd_solution)
# (g_sol,) = g(pgd_solution)
# print('value = ', f_sol + g_sol)


# pgd_solution = pgd(f=f, g=g, initial_value=initial_value,
#                         stepsize=.5, reg_parameter= 0.01,
#                         max_iterations=200,
#                         backtrack_factor=.75)

# plt.imshow(pgd_solution[0,0,0].abs())
# plt.colorbar()
# (f_sol,) = f(pgd_solution)
# (g_sol,) = g(pgd_solution)
# print('value = ', f_sol + g_sol)