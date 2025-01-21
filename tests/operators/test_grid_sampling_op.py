"""Tests for grid sampling operator."""

import contextlib

import pytest
import torch
from mrpro.data import SpatialDimension
from mrpro.operators import GridSamplingOp
from torch.autograd.gradcheck import gradcheck

from tests import RandomGenerator, dotproduct_adjointness_test


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
    # bicubic only supports 2D
    _test_grid_sampling_op_adjoint(dim=2, interpolation_mode=interpolation_mode)


@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
def test_grid_sampling_op_padding_mode(padding_mode):
    """Test for different padding_modes."""
    _test_grid_sampling_op_adjoint(padding_mode=padding_mode)


@pytest.mark.parametrize('align_corners', ['no_align', 'align'])
def test_grid_sampling_op_align_mode(align_corners):
    """Test for different align modes ."""
    _test_grid_sampling_op_adjoint(align_corners=align_corners)


def _test_grid_sampling_op_adjoint(
    dtype='float32',
    dim=2,
    interpolation_mode='bilinear',
    padding_mode='zeros',
    align_corners='no_align',
    batched='non_batched',
    channel='single_channel',
):
    """Used in the tests above."""
    rng = getattr(RandomGenerator(0), f'{dtype}_tensor')
    batch = (2, 3) if batched == 'batched' else (1,)
    channel = (5, 6) if channel == 'multi_channel' else (1,)
    align_corners_bool = align_corners == 'align'
    zyx_v = (7, 8, 9)[-dim:]
    zyx_u = (11, 12, 13)[-dim:]
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
    """Used in the tests above."""
    rng = RandomGenerator(0).float32_tensor
    batch = (2, 3)
    channel = (5, 7)
    zyx_v = (7, 10, 20)[-dim:]
    zyx_u = (9, 22, 30)[-dim:]
    grid = rng((*batch, *zyx_v, dim), -1, 1.0)
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
    """Gradient check for forward wrt x."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8)
    u = rng((1, 1, 3, 5)).requires_grad_(True)
    gradcheck(lambda grid, u: GridSamplingOp(grid, input_shape=SpatialDimension(1, 3, 5))(u), (grid, u), fast_mode=True)


def test_grid_sampling_op_gradcheck_grid_forward():
    """Gradient check for forward wrt grid."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    u = rng((1, 1, 3, 5))
    gradcheck(lambda grid, u: GridSamplingOp(grid, input_shape=SpatialDimension(1, 3, 5))(u), (grid, u), fast_mode=True)


def test_grid_sampling_op_gradcheck_x_adjoint():
    """Gradient check for adjoint wrt x."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8)
    v = rng((2, 1, 1, 2)).requires_grad_(True)
    gradcheck(
        lambda grid, v: GridSamplingOp(grid, input_shape=SpatialDimension(1, 2, 3)).adjoint(v),
        (grid, v),
        fast_mode=True,
    )


def test_grid_sampling_op_gradcheck_grid_adjoint():
    """Gradient check for adjoint wrt grid."""
    rng = RandomGenerator(0).float64_tensor
    grid = rng((2, 1, 2, 2), -0.8, 0.8).requires_grad_(True)
    v = rng((2, 1, 1, 2))
    gradcheck(
        lambda grid, v: GridSamplingOp(grid, input_shape=SpatialDimension(1, 2, 3)).adjoint(v),
        (grid, v),
        fast_mode=True,
    )


def test_grid_sampling_op_errormsg_gridlastdim():
    """Test if error message on wrong last dim is raised."""
    grid = torch.ones(1, 2, 3, 4)
    with pytest.raises(ValueError, match='last dimension'):
        _ = GridSamplingOp(grid, SpatialDimension(1, 1, 1))


def test_grid_sampling_op_errormsg_gridndims_3d():
    """Test if error message on missing batch dim is raised."""
    grid = torch.ones(1, 1, 1, 3)
    with pytest.raises(ValueError, match='batch z y x 3'):
        _ = GridSamplingOp(grid, SpatialDimension(1, 1, 1))


def test_grid_sampling_op_errormsg_gridndims_2d():
    """Test if error message on missing batch dim is raised."""
    grid = torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match='batch y x 2'):
        _ = GridSamplingOp(grid, SpatialDimension(1, 1, 1))


def test_grid_sampling_op_errormsg_cubic3d():
    """Test if error for 3D cubic is raised."""
    grid = torch.ones(1, 1, 1, 1, 3)  # 3d
    with pytest.raises(NotImplementedError, match='cubic'):
        _ = GridSamplingOp(grid, SpatialDimension(1, 1, 1), interpolation_mode='bicubic')


def test_grid_sampling_op_errormsg_complexgrid():
    """Test if error for complex grid is raised."""
    grid = torch.ones(1, 1, 1, 1, 3) + 0j
    with pytest.raises(ValueError, match='real'):
        _ = GridSamplingOp(grid, SpatialDimension(1, 1, 1))


@pytest.mark.parametrize(
    ('value', 'error_message'),
    [(1.0001, 'values outside range'), (-1.0001, 'values outside range'), (1.0, None), (-1.0, None)],
)
def test_grid_sampling_op_warning_gridrange(value, error_message):
    """Test if warning for grid values outside [-1,1] is raised"""
    grid = torch.zeros(1, 1, 1, 1, 3)
    grid[..., 1] = value
    conditional_warn: contextlib.AbstractContextManager[None] | pytest.WarningsRecorder = (
        pytest.warns(UserWarning, match=error_message) if error_message else contextlib.nullcontext()
    )
    with conditional_warn:
        _ = GridSamplingOp(grid, SpatialDimension(1, 1, 1))


def test_grid_sampling_op_errormsg_inputdim_3d():
    """Test if error for wrong input dimensions is raised."""
    grid = torch.ones(1, 1, 1, 1, 3)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid, input_shape)
    u = torch.zeros(1, 2, 3, 4)
    with pytest.raises(ValueError, match='5 dimensions: batch channel z y x'):
        _ = operator(u)


def test_grid_sampling_op_warningmsg_inputshape_3d():
    """Test if warning for wrong input_shape is raised in forward"""
    grid = torch.ones(1, 1, 1, 1, 3)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid, input_shape)
    u = torch.zeros(1, 1, 3, 3, 4)
    with pytest.warns(UserWarning, match='Mismatch'):
        _ = operator(u)


def test_grid_sampling_op_errormsg_inputdim_2d():
    """Test if error for wrong input dimensions is raised."""
    grid = torch.ones(1, 1, 1, 2)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid, input_shape)
    u = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError, match='4 dimensions: batch channel y x'):
        _ = operator(u)


def test_grid_sampling_op_warningmsg_inputshape_2d():
    """Test if warning for wrong input_shape is raised in forward"""
    grid = torch.ones(1, 1, 1, 2)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid, input_shape)
    u = torch.zeros(1, 2, 3, 5)
    with pytest.warns(UserWarning, match='Mismatch'):
        _ = operator(u)


def test_grid_sampling_op_errormsg_inputdim_z_2d():
    """Test if no error for wrong input dimensions is raised if only z is wrong for 2d."""
    grid = torch.ones(1, 1, 1, 2)
    input_shape = SpatialDimension(2, 3, 4)
    operator = GridSamplingOp(grid, input_shape)
    u = torch.zeros(1, 17, 3, 4)
    _ = operator(u)  # works, as z is ignored.


@pytest.mark.parametrize(
    ('grid_batch', 'u_batch', 'channel', 'expected_output'),
    [
        ((1,), (1,), (1,), (1, 1)),
        ((7, 1, 2), (1, 8, 2), (2, 3), (7, 8, 2, 2, 3)),
        ((3,), (4,), (1,), 'not broadcastable'),
        ((7, 1, 2), (1, 1, 2), (4,), (7, 1, 2, 4)),
        ((7, 1, 2), (2,), (4,), 'not broadcastable'),
    ],
)
def test_grid_sampling_op_batchdims(grid_batch, u_batch, channel, expected_output):
    """Test if error for wrong input dimensions is raised."""
    grid = torch.ones(*grid_batch, 7, 8, 9, 3)  # 3d
    input_shape = SpatialDimension(2, 3, 4)
    u = torch.zeros(*u_batch, *channel, *input_shape.zyx)
    operator = GridSamplingOp(grid, input_shape)
    if isinstance(expected_output, str):
        with pytest.raises(ValueError, match=expected_output):
            _ = operator(u)
    else:
        (result,) = operator(u)
        assert result.shape == (*expected_output, 7, 8, 9)
