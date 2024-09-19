"""Tests for Zero Pad Operator class."""

import pytest
import torch
from mrpro.operators import ZeroPadOp

from tests import RandomGenerator
from tests.helper import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_zero_pad_op_and_domain_range(u_shape, v_shape):
    """Create a zero padding operator and an element from domain and range."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(u_shape)
    v = generator.complex64_tensor(v_shape)
    zero_padding_op = ZeroPadOp(dim=(-3, -2, -1), original_shape=u_shape, padded_shape=v_shape)
    return zero_padding_op, u, v


def test_zero_pad_op_content():
    """Test correct padding and cropping (i.e. negative padding size)."""
    original_shape = (2, 100, 3, 200, 50, 2)
    padded_shape = (2, 80, 3, 100, 240, 2)
    generator = RandomGenerator(seed=0)
    original_data = generator.complex64_tensor(original_shape)
    padding_dimensions = (-5, -3, -2)
    zero_padding_op = ZeroPadOp(
        dim=padding_dimensions,
        original_shape=tuple([original_shape[d] for d in padding_dimensions]),
        padded_shape=tuple([padded_shape[d] for d in padding_dimensions]),
    )
    (padded_data,) = zero_padding_op.forward(original_data)

    # Compare overlapping region
    torch.testing.assert_close(original_data[:, 10:90, :, 50:150, :, :], padded_data[:, :, :, :, 95:145, :])


@pytest.mark.parametrize(
    ('u_shape', 'v_shape'),
    [
        ((101, 201, 50), (13, 221, 64)),
        ((100, 200, 50), (14, 220, 64)),
        ((101, 201, 50), (14, 220, 64)),
        ((100, 200, 50), (13, 221, 64)),
    ],
)
def test_zero_pad_op_adjoint(u_shape, v_shape):
    """Test adjointness of pad operator."""
    dotproduct_adjointness_test(*create_zero_pad_op_and_domain_range(u_shape, v_shape))


def test_zero_pad_op_grad():
    """Test gradient of zero padding operator."""
    gradient_of_linear_operator_test(*create_zero_pad_op_and_domain_range((101, 201, 50), (13, 221, 64)))


def test_zero_pad_op_forward_mode_autodiff():
    """Test forward-mode autodiff of zero padding operator."""
    forward_mode_autodiff_of_linear_operator_test(*create_zero_pad_op_and_domain_range((101, 201, 50), (13, 221, 64)))
