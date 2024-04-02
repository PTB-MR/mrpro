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
from mrpro.operators import GridSamplingOp
from torch.autograd.gradcheck import gradcheck

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64'])
def test_grid_sampling_op_dtype(dtype):
    """Test for different data types."""
    _test_grid_sampling_op_adjoint(dtype=dtype)


@pytest.mark.parametrize('dim_str', ['2D', '3D'])
@pytest.mark.parametrize('batched', ['batched', 'non_batched'])
@pytest.mark.parametrize('channel', ['multi_channel', 'single_channel'])
def test_grid_sampling_op_dim_batch_channel(dim_str, batched, channel):
    """Test for different dimensions."""
    _test_grid_sampling_op_adjoint(dim=int(dim_str[0]), batched=batched, channel=channel)


@pytest.mark.parametrize('interpolation_mode', ['bilinear', 'nearest', 'bicubic'])
def test_grid_sampling_op_interpolation_mode(interpolation_mode):
    """Test for different interpolation_modes."""
    _test_grid_sampling_op_adjoint(dim=2, interpolation_mode=interpolation_mode)


@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
def test_grid_sampling_op_padding_mode(padding_mode):
    """Test for different padding_modes."""
    _test_grid_sampling_op_adjoint(padding_mode=padding_mode)


@pytest.mark.parametrize('align_corners', ['no_align', 'align'])
def test_grid_sampling_op_align_mode(align_corners):
    """Test for different align modes ."""
    _test_grid_sampling_op_adjoint(align_corners=align_corners == 'align')


def _test_grid_sampling_op_adjoint(
    dtype='float32',
    dim=2,
    interpolation_mode='bilinear',
    padding_mode='zeros',
    align_corners='no_align',
    batched='non_batched',
    channel='single_channel',
):
    rng = getattr(RandomGenerator(0), f'{dtype}_tensor')
    batch = (2, 3) if batched == 'batched' else (1,)
    channel = (5, 6) if channel == 'multi_channel' else (1,)
    align_corners_bool = align_corners == 'align'
    zyx_v = (7, 8, 9)[:dim]
    zyx_u = (11, 12, 13)[:dim]
    grid = RandomGenerator(42).float64_tensor((*batch, *zyx_v, dim), -1, 1)
    input_shape = SpatialDimension(z=(99 if dim == 2 else zyx_u[-3]), y=zyx_u[-2], x=zyx_u[-1])
    operator = GridSamplingOp(
        grid,
        input_shape=input_shape,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners_bool,
    )
    operator = operator.to(dtype=getattr(torch, dtype).to_real())
    u = rng((*batch, *channel, *zyx_u))
    v = rng((*batch, *channel, *zyx_v))
    dotproduct_adjointness_test(operator, u, v)


@pytest.mark.parametrize('interpolation_mode', ['bilinear', 'nearest', 'bicubic'])
def test_grid_sampling_op_interpolation_mode_backward_is_adjoint(interpolation_mode):
    """Test for different interpolation_modes."""
    # bicubic only supports 2D
    dim = 2 if interpolation_mode == 'bicubic' else 3
    _test_grid_sampling_op_x_backward(dim=dim, interpolation_mode=interpolation_mode)


@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
def test_grid_sampling_op_padding_mode_backward_is_adjoint(padding_mode):
    """Test for different padding_modes."""
    _test_grid_sampling_op_x_backward(padding_mode=padding_mode)


@pytest.mark.parametrize('align_corners', ['no_align', 'align'])
def test_grid_sampling_op_align_mode_backward_is_adjoint(align_corners):
    """Test for different align modes ."""
    _test_grid_sampling_op_x_backward(align_corners=align_corners == 'align')


def _test_grid_sampling_op_x_backward(dim=3, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
    rng = RandomGenerator(0).float32_tensor
    batch = (2, 3)
    channel = (5, 7)
    zyx_v = (7, 10, 20)[:dim]
    zyx_u = (9, 22, 30)[:dim]
    grid = rng((*batch, *zyx_v, 2), -1, 1.0)
    input_shape = SpatialDimension(z=99 if dim == 2 else zyx_u[-3], y=zyx_u[-2], x=zyx_u[-1])
    operator = GridSamplingOp(
        grid,
        input_shape=input_shape,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    u = rng((*batch, *channel, *zyx_u)).requires_grad_(True)
    v = rng((*batch, *channel, *zyx_v)).requires_grad_(True)
    (forward_u,) = operator(u)
    forward_u.backward(v.detach())
    (adjoint_v,) = operator.adjoint(v)
    adjoint_v.backward(u.detach())
    torch.testing.assert_close(u.grad, adjoint_v)
    torch.testing.assert_close(v.grad, forward_u)


def test_grid_sampling_op_gradcheck_x_forward():
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    u = rng((1, 1, 3, 5)).requires_grad_(True)
    gradcheck(lambda grid, u: GridSamplingOp(grid, input_shape=SpatialDimension(1, 3, 5))(u), (grid, u), fast_mode=True)


def test_grid_sampling_op_gradcheck_grid_forward():
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    u = rng((1, 1, 3, 5))
    gradcheck(lambda grid, u: GridSamplingOp(grid, input_shape=SpatialDimension(1, 3, 5))(u), (grid, u), fast_mode=True)


def test_grid_sampling_op_gradcheck_x_adjoint():
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    v = rng((2, 1, 1, 2))
    gradcheck(
        lambda grid, v: GridSamplingOp(grid, input_shape=SpatialDimension(1, 2, 3)).adjoint(v),
        (grid, v),
        fast_mode=True,
    )


def test_grid_sampling_op_gradcheck_grid_adjoint():
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    v = rng((2, 1, 1, 2))
    gradcheck(
        lambda grid, v: GridSamplingOp(grid, input_shape=SpatialDimension(1, 2, 3)).adjoint(v),
        (grid, v),
        fast_mode=True,
    )
