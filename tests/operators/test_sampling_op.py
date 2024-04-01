"""Tests for sensitivity operator."""

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

import pytest
import torch
from mrpro.data import SpatialDimension
from mrpro.operators._SamplingOp import SamplingOp
from torch.autograd.gradcheck import gradcheck

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64'])
def test_sampling_op(dtype):
    grid = torch.stack(torch.meshgrid(torch.arange(10).float(), torch.arange(20).float(), indexing='ij'), -1)[None]
    grid = torch.stack((grid, grid, grid))
    operator = SamplingOp(grid, input_shape=SpatialDimension(1, 22, 30)).to(dtype=getattr(torch, dtype).to_real())
    rng = getattr(RandomGenerator(0), f'{dtype}_tensor')
    u = rng((3, 5, 7, 11, 22, 30))
    v = rng((3, 5, 7, 11, 10, 20))
    dotproduct_adjointness_test(operator, u, v)


def test_sampling_op_x_backward():
    grid = torch.stack(torch.meshgrid(torch.arange(10).float(), torch.arange(20).float(), indexing='ij'), -1)[None]
    grid = torch.stack((grid, grid, grid))
    operator = SamplingOp(grid, input_shape=SpatialDimension(1, 22, 30))
    rng = RandomGenerator(0).float32_tensor
    u = rng((3, 5, 7, 11, 22, 30)).requires_grad_(True)
    v = rng((3, 5, 7, 11, 10, 20)).requires_grad_(True)
    (forward_u,) = operator(u)
    forward_u.backward(v.detach())
    (adjoint_v,) = operator.adjoint(v)
    adjoint_v.backward(u.detach())
    torch.testing.assert_close(u.grad, adjoint_v)
    torch.testing.assert_close(v.grad, forward_u)


def test_sampling_op_gradcheck_x_forward():
    grid = torch.stack(torch.meshgrid(torch.arange(3).float(), torch.arange(2).float(), indexing='ij'), -1)[None]
    grid = grid.double()
    rng = RandomGenerator(0).float64_tensor
    u = rng((1, 1, 3, 5)).requires_grad_(True)
    gradcheck(lambda grid, u: SamplingOp(grid, input_shape=SpatialDimension(1, 3, 5))(u), (grid, u))


def test_sampling_op_gradcheck_grid_forward():
    grid = torch.stack(torch.meshgrid(torch.arange(3).float(), torch.arange(2).float(), indexing='ij'), -1)[None]
    grid = grid.double().requires_grad_(True)
    rng = RandomGenerator(0).float64_tensor
    u = rng((1, 1, 3, 5))
    gradcheck(lambda grid, u: SamplingOp(grid, input_shape=SpatialDimension(1, 3, 5))(u), (grid, u))


def test_sampling_op_gradcheck_x_adjoint():
    grid = torch.stack(torch.meshgrid(torch.arange(3).float(), torch.arange(2).float(), indexing='ij'), -1)[None]
    grid = grid.double()
    rng = RandomGenerator(0).float64_tensor
    v = rng((1, 1, 3, 2)).requires_grad_(True)
    gradcheck(lambda grid, v: SamplingOp(grid, input_shape=SpatialDimension(1, 3, 5)).adjoint(v), (grid, v))


def test_sampling_op_gradcheck_grid_adjoint():
    grid = torch.stack(torch.meshgrid(torch.arange(3).float(), torch.arange(2).float(), indexing='ij'), -1)[None]
    grid = grid.double().requires_grad_(True)
    rng = RandomGenerator(0).float64_tensor
    v = rng((1, 1, 3, 2))
    gradcheck(lambda grid, v: SamplingOp(grid, input_shape=SpatialDimension(1, 3, 5)).adjoint(v), (grid, v))
