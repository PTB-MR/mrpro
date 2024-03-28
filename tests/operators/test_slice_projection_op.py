"""Tests for projection operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch
from mrpro.data import SpatialDimension
from mrpro.operators import SliceProjectionOp
from mrpro.operators._SliceProjectionOp import SliceGaussian
from mrpro.operators._SliceProjectionOp import SliceInterpolate
from mrpro.operators._SliceProjectionOp import SliceSmoothedRect
from mrpro.utils import Rotation

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def test_slice_projection_op_basic():
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = None
    slice_shift = 0.0
    slice_profile = 1.0
    operator = SliceProjectionOp(
        input_shape=input_shape, slice_rotation=slice_rotation, slice_shift=slice_shift, slice_profile=slice_profile
    )
    volume = torch.ones(input_shape.zyx)
    (slice2d,) = operator(volume)
    assert slice2d.shape == (1, 1, 30, 30)
    expected = torch.zeros(1, 1, 30, 30)
    expected[:, :, 5:-5, :] = 1
    torch.testing.assert_close(slice2d, expected)


@pytest.mark.parametrize('dtype', ['complex64', 'float64', 'float32'])
@pytest.mark.parametrize('optimize_for', ['forward', 'adjoint', 'both'])
def test_slice_projection_op_basic_adjointness(optimize_for, dtype):
    rng = getattr(RandomGenerator(314), f'{dtype}_tensor')
    operator_dtype = getattr(torch, dtype).to_real()
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = None
    slice_shift = 0.0
    slice_profile = 1.0
    operator = SliceProjectionOp(
        input_shape=input_shape,
        slice_rotation=slice_rotation,
        slice_shift=slice_shift,
        slice_profile=slice_profile,
        optimize_for=optimize_for,
    )
    operator = operator.to(operator_dtype)
    u = rng((1, *input_shape.zyx))
    v = rng((1, 1, 1, max(input_shape.zyx), max(input_shape.zyx)))
    dotproduct_adjointness_test(operator, u, v)


def test_slice_projection_op_slice_batching():
    rng = RandomGenerator(314).float32_tensor
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = Rotation.random((5, 1), 0)
    slice_shift = rng((5, 3))
    xp = torch.linspace(-2, 2, 100)
    yp = (xp.abs() < 1).float()
    interpolated_profile = SliceInterpolate(xp, yp)
    slice_profile = np.array([SliceGaussian(1.0), SliceSmoothedRect(1.0, 1.0), interpolated_profile])[None, :]
    operator = SliceProjectionOp(
        input_shape=input_shape,
        slice_rotation=slice_rotation,
        slice_shift=slice_shift,
        slice_profile=slice_profile,
    )
    u = rng(input_shape.zyx)
    v = rng((5, 3, 1, max(input_shape.zyx), max(input_shape.zyx)))
    dotproduct_adjointness_test(operator, u, v)


def test_slice_projection_op_volume_batching():
    rng = RandomGenerator(314).float32_tensor
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = None
    slice_shift = rng(3)
    slice_profile = 1.0
    operator = SliceProjectionOp(
        input_shape=input_shape,
        slice_rotation=slice_rotation,
        slice_shift=slice_shift,
        slice_profile=slice_profile,
    )
    u = rng((5, *input_shape.zyx))
    v = rng((3, 5, 1, max(input_shape.zyx), max(input_shape.zyx)))
    dotproduct_adjointness_test(operator, u, v)


@pytest.mark.parametrize('direction', ['forward', 'adjoint'])
@pytest.mark.parametrize('dtype', ['complex64', 'float64', 'float32'])
@pytest.mark.parametrize('optimize_for', ['forward', 'adjoint', 'both'])
def test_slice_projection_op_backward_is_adjoint(optimize_for, dtype, direction):
    rng = getattr(RandomGenerator(314), f'{dtype}_tensor')
    operator_dtype = getattr(torch, dtype).to_real()
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = None
    slice_shift = 0.0
    slice_profile = 1.0
    operator = SliceProjectionOp(
        input_shape=input_shape,
        slice_rotation=slice_rotation,
        slice_shift=slice_shift,
        slice_profile=slice_profile,
        optimize_for=optimize_for,
    )
    operator = operator.to(operator_dtype)
    u = rng(input_shape.zyx).requires_grad_(True)
    v = rng((1, 1, max(input_shape.zyx), max(input_shape.zyx))).requires_grad_(True)
    match direction:
        case 'forward':  # backward of forward
            (forward_u,) = operator(u)
            forward_u.backward(v)
            adjoint_v = u.grad
        case 'adjoint':  # backward of adjoint
            (adjoint_v,) = operator.adjoint(v)
            adjoint_v.backward(u)
            forward_u = v.grad

    assert forward_u.shape == v.shape
    assert adjoint_v.shape == u.shape
    dotproduct_range = torch.vdot(forward_u.flatten(), v.flatten())
    dotproduct_domain = torch.vdot(u.flatten().flatten(), adjoint_v.flatten())
    torch.testing.assert_close(dotproduct_range, dotproduct_domain)
