"""Tests for projection operator."""

import math
from typing import Literal

import numpy as np
import pytest
import torch
from mrpro.data import Rotation, SpatialDimension
from mrpro.operators import SliceProjectionOp
from mrpro.utils import RandomGenerator
from mrpro.utils.slice_profiles import SliceGaussian, SliceInterpolate, SliceSmoothedRectangular

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_slice_projection_op_and_domain_range(
    optimize_for: Literal['forward', 'adjoint', 'both'], dtype: torch.dtype
) -> tuple[SliceProjectionOp, torch.Tensor, torch.Tensor]:
    """Create a slice projection operator and an element from domain and range."""
    rng = RandomGenerator(314)
    operator_dtype = dtype.to_real()
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
    u = rng.rand_tensor((1, *input_shape.zyx), dtype=dtype)
    v = rng.rand_tensor((1, 1, 1, max(input_shape.zyx), max(input_shape.zyx)), dtype=dtype)
    return operator, u, v


def test_slice_projection_op_cube_basic() -> None:
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


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_slice_projection_op_cube_rotation(axis: Literal['x', 'y', 'z']) -> None:
    input_shape = SpatialDimension(201, 201, 201)
    slice_rotation = Rotation.from_euler(axis, 45, degrees=True)
    slice_shift = 0.0
    slice_profile = 1.0
    operator = SliceProjectionOp(
        input_shape=input_shape, slice_rotation=slice_rotation, slice_shift=slice_shift, slice_profile=slice_profile
    )
    volume = torch.zeros(input_shape.zyx)
    volume[50:151, 50:151, 50:151] = 1  # 101 x 101 x 101 cube
    (slice2d,) = operator(volume)

    if axis == 'z':
        # for a 45 degree rotation in z, we should see a 45 rotated square in the slice
        # the diagonal should be in the center and 101*sqrt(2) pixels long
        assert (slice2d > 0.01).sum(-1).max() == math.ceil(101 * 2**0.5)
        assert torch.argmax((slice2d > 0.01).sum(-1)) == 100
        assert (slice2d > 0.01).sum(-2).max() == math.ceil(101 * 2**0.5)
        assert torch.argmax((slice2d > 0.01).sum(-2)) == 100
        # and the area should be 101**2
        torch.testing.assert_close(slice2d.sum(), torch.tensor(101.0) ** 2, atol=0, rtol=1e-3)
    else:
        # for a 45 degree rotation in x or y, we should see one diagonal of the cube in the slice
        # rotation around x should show the diagonal in the y direction and vice versa
        diagonal_dir = -1 if axis == 'y' else -2
        assert (slice2d > 0).sum(diagonal_dir).max() == math.ceil(101 * 2**0.5)
        assert (slice2d > 0).sum() == round((2**0.5) * 101) * 101


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
@pytest.mark.parametrize('match_shift', [True, False])
def test_slice_projection_op_cube_shift(axis: Literal['x', 'y', 'z'], match_shift: bool) -> None:
    input_shape = SpatialDimension(21, 21, 21)
    slice_profile = 1.0
    slice_rotation = Rotation.from_euler(axis, 90, degrees=True)
    shift = SpatialDimension(1, 2, 3)
    volume = torch.zeros(input_shape.zyx)
    # single pixel is marked in the volume
    volume[10 + shift.z, 10 + shift.y, 10 + shift.x] = 1

    # shift the slice by the same amount as the pixel in the volume
    if axis == 'x':  # rotation around x
        # 90 degree rotation around x and shift moves along y
        slice_shift = +shift.y if match_shift else 0.0
    if axis == 'y':  # rotation around y
        # 90 degree rotation around y and shift moves along -x
        slice_shift = -shift.x if match_shift else 0.0
    if axis == 'z':  # rotation around z
        slice_shift = shift.z if match_shift else 0.0

    operator = SliceProjectionOp(
        input_shape=input_shape, slice_rotation=slice_rotation, slice_shift=slice_shift, slice_profile=slice_profile
    )
    (slice2d,) = operator(volume)

    # did we find the pixel in the slice?
    assert (slice2d > 0.01).sum() == match_shift


def test_slice_projection_width_error() -> None:
    input_shape = SpatialDimension(1, 1, 1)
    slice_profile = 0.99
    with pytest.raises(ValueError, match='width'):
        _ = SliceProjectionOp(input_shape=input_shape, slice_profile=slice_profile)


@pytest.mark.parametrize(
    'dtype', [torch.complex64, torch.float64, torch.float32], ids=['complex64', 'float64', 'float32']
)
@pytest.mark.parametrize('optimize_for', ['forward', 'adjoint', 'both'])
def test_slice_projection_op_basic_adjointness(
    optimize_for: Literal['forward', 'adjoint', 'both'], dtype: torch.dtype
) -> None:
    dotproduct_adjointness_test(*create_slice_projection_op_and_domain_range(optimize_for, dtype))


def test_slice_projection_op_slice_batching() -> None:
    rng = RandomGenerator(314).float32_tensor
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = Rotation.random((5, 1), 0)
    slice_shift = rng((5, 3))
    xp = torch.linspace(-2, 2, 100)
    yp = (xp.abs() < 1).float()
    interpolated_profile = SliceInterpolate(xp, yp)
    slice_profile = np.array([SliceGaussian(1.0), SliceSmoothedRectangular(1.0, 1.0), interpolated_profile])[None, :]
    operator = SliceProjectionOp(
        input_shape=input_shape,
        slice_rotation=slice_rotation,
        slice_shift=slice_shift,
        slice_profile=slice_profile,
    )
    u = rng(input_shape.zyx)
    v = rng((5, 3, 1, max(input_shape.zyx), max(input_shape.zyx)))
    dotproduct_adjointness_test(operator, u, v)


def test_slice_projection_op_volume_batching() -> None:
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
@pytest.mark.parametrize(
    'dtype', [torch.complex64, torch.float64, torch.float32], ids=['complex64', 'float64', 'float32']
)
@pytest.mark.parametrize('optimize_for', ['forward', 'adjoint', 'both'])
def test_slice_projection_op_backward_is_adjoint(
    optimize_for: Literal['forward', 'adjoint', 'both'], dtype: torch.dtype, direction: str
) -> None:
    rng = RandomGenerator(314)
    operator_dtype = dtype.to_real()
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
    u = rng.rand_tensor(input_shape.zyx, dtype=dtype).requires_grad_(True)
    v = rng.rand_tensor((1, 1, max(input_shape.zyx), max(input_shape.zyx)), dtype=dtype).requires_grad_(True)
    match direction:
        case 'forward':  # backward of forward
            (forward_u,) = operator(u)
            forward_u.backward(v)
            assert u.grad is not None
            adjoint_v = u.grad
        case 'adjoint':  # backward of adjoint
            (adjoint_v,) = operator.adjoint(v)
            adjoint_v.backward(u)
            assert v.grad is not None
            forward_u = v.grad
    assert forward_u.shape == v.shape
    assert adjoint_v.shape == u.shape
    dotproduct_range = torch.vdot(forward_u.flatten(), v.flatten())
    dotproduct_domain = torch.vdot(u.flatten().flatten(), adjoint_v.flatten())
    torch.testing.assert_close(dotproduct_range, dotproduct_domain)


# Sparse tensors do currently not work with torch.func. See https://github.com/pytorch/pytorch/issues/136357
@pytest.mark.parametrize(
    'dtype', [torch.complex64, torch.float64, torch.float32], ids=['complex64', 'float64', 'float32']
)
@pytest.mark.parametrize('optimize_for', ['forward', 'adjoint', 'both'])
def test_slice_projection_op_grad(optimize_for: Literal['forward', 'adjoint', 'both'], dtype: torch.dtype) -> None:
    """Test gradient of slice projection operator."""
    with pytest.raises(RuntimeError, match='Sparse CSR tensors'):
        gradient_of_linear_operator_test(*create_slice_projection_op_and_domain_range(optimize_for, dtype))


@pytest.mark.parametrize(
    'dtype', [torch.complex64, torch.float64, torch.float32], ids=['complex64', 'float64', 'float32']
)
@pytest.mark.parametrize('optimize_for', ['forward', 'adjoint', 'both'])
def test_slice_projection_op_forward_mode_autodiff(
    optimize_for: Literal['forward', 'adjoint', 'both'], dtype: torch.dtype
) -> None:
    """Test forward-mode autodiff of slice projection operator."""
    with pytest.raises(RuntimeError, match='Sparse CSR tensors'):
        forward_mode_autodiff_of_linear_operator_test(*create_slice_projection_op_and_domain_range(optimize_for, dtype))


@pytest.mark.cuda
def test_slice_projection_op_cuda() -> None:
    """Test slice projection operator works on CUDA devices."""
    rng = RandomGenerator(0)
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = Rotation.random(4)
    slice_shift = rng.float32_tensor(4)
    slice_profile = SliceGaussian(1.0)
    u = rng.float32_tensor(input_shape.zyx)

    # Create on CPU, transfer to GPU, run on GPU
    sliceprojection_op = SliceProjectionOp(
        input_shape=input_shape,
        slice_rotation=slice_rotation,
        slice_shift=slice_shift,
        slice_profile=slice_profile,
        optimize_for='forward',
    )
    operator = sliceprojection_op.H @ sliceprojection_op
    operator.cuda()
    (result,) = operator(u.cuda())
    assert result.is_cuda

    # Creation on GPU is not supported (see docstring)
