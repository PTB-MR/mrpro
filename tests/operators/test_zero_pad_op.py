"""Tests for Zero Pad Operator class."""

import pytest
import torch
from mrpro.operators import ZeroPadOp
from typing_extensions import Unpack

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_zero_pad_op_and_domain_range(
    u_shape: tuple[int, int, int, Unpack[tuple[int, ...]]], v_shape: tuple[int, int, int, Unpack[tuple[int, ...]]]
) -> tuple[ZeroPadOp, torch.Tensor, torch.Tensor]:
    """Create a zero padding operator and an element from domain and range."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(u_shape)
    v = generator.complex64_tensor(v_shape)
    zero_padding_op = ZeroPadOp(dim=(-3, -2, -1), original_shape=u_shape, padded_shape=v_shape)
    return zero_padding_op, u, v


def test_zero_pad_op_content() -> None:
    """Test correct padding and cropping (i.e. negative padding size)."""
    original_shape = (2, 30, 3, 40, 50, 2)
    padded_shape = (2, 10, 3, 30, 80, 2)
    generator = RandomGenerator(seed=0)
    original_data = generator.complex64_tensor(original_shape)
    padding_dimensions = (-5, -3, -2)
    zero_padding_op = ZeroPadOp(
        dim=padding_dimensions,
        original_shape=tuple([original_shape[d] for d in padding_dimensions]),
        padded_shape=tuple([padded_shape[d] for d in padding_dimensions]),
    )
    (padded_data,) = zero_padding_op(original_data)

    # Compare overlapping region
    torch.testing.assert_close(original_data[:, 10:20, :, 5:35, :, :], padded_data[:, :, :, :, 15:65, :])


SHAPE_PARAMETERS = pytest.mark.parametrize(
    ('u_shape', 'v_shape'),
    [
        ((101, 201, 50), (13, 221, 64)),
        ((100, 200, 50), (14, 220, 64)),
        ((101, 201, 50), (14, 220, 64)),
        ((100, 200, 50), (13, 221, 64)),
    ],
)


@SHAPE_PARAMETERS
def test_zero_pad_op_adjoint(
    u_shape: tuple[int, int, int, Unpack[tuple[int, ...]]], v_shape: tuple[int, int, int, Unpack[tuple[int, ...]]]
) -> None:
    """Test adjointness of pad operator."""
    dotproduct_adjointness_test(*create_zero_pad_op_and_domain_range(u_shape, v_shape))


@SHAPE_PARAMETERS
def test_zero_pad_op_grad(
    u_shape: tuple[int, int, int, Unpack[tuple[int, ...]]], v_shape: tuple[int, int, int, Unpack[tuple[int, ...]]]
) -> None:
    """Test gradient of zero padding operator."""
    gradient_of_linear_operator_test(*create_zero_pad_op_and_domain_range(u_shape, v_shape))


@SHAPE_PARAMETERS
def test_zero_pad_op_forward_mode_autodiff(
    u_shape: tuple[int, int, int, Unpack[tuple[int, ...]]], v_shape: tuple[int, int, int, Unpack[tuple[int, ...]]]
) -> None:
    """Test forward-mode autodiff of zero padding operator."""
    forward_mode_autodiff_of_linear_operator_test(*create_zero_pad_op_and_domain_range(u_shape, v_shape))


@pytest.mark.cuda
def test_zero_pad_op_cuda() -> None:
    """Test ZeroPadOp works on CUDA devices."""
    generator = RandomGenerator(seed=0)
    original_shape = (101, 201, 50)
    padded_shape = (13, 221, 64)
    padding_dim = (-3, -2, -1)
    x = generator.complex64_tensor(original_shape)

    # Create on CPU, run on CPU
    zero_padding_op = ZeroPadOp(dim=padding_dim, original_shape=original_shape, padded_shape=padded_shape)
    (x_padded,) = zero_padding_op(x)
    assert x_padded.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    zero_padding_op = ZeroPadOp(dim=padding_dim, original_shape=original_shape, padded_shape=padded_shape)
    zero_padding_op.cuda()
    (x_padded,) = zero_padding_op(x.cuda())
    assert x_padded.is_cuda
