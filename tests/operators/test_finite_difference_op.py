"""Tests for finite difference operator."""

from collections.abc import Sequence
from typing import Literal

import pytest
import torch
from einops import repeat
from mrpro.operators import FiniteDifferenceOp

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_finite_difference_op_and_range_domain(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> tuple[FiniteDifferenceOp, torch.Tensor, torch.Tensor]:
    """Create a finite difference operator and an element from domain and range."""
    random_generator = RandomGenerator(seed=0)
    img_shape = (5, 6, 4, 10, 20, 16)

    # Generate finite difference operator
    finite_difference_op = FiniteDifferenceOp(dim, mode, pad_mode)

    u = random_generator.complex64_tensor(size=img_shape)
    v = random_generator.complex64_tensor(size=(len(dim), *img_shape))
    return finite_difference_op, u, v


@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
def test_finite_difference_op_forward(mode: Literal['central', 'forward', 'backward']) -> None:
    """Test correct finite difference of simple object."""
    # Test object with positive linear gradient in real and negative linear gradient imaginary part
    linear_gradient_object = (
        repeat(torch.arange(1, 21), 'x -> y x', y=1) + 2 * repeat(torch.arange(1, 21), 'y -> y x', x=1)
    ).to(dtype=torch.float32)
    linear_gradient_object = linear_gradient_object - 1j * linear_gradient_object

    # Generate and apply finite difference operator
    finite_difference_op = FiniteDifferenceOp(dim=(-1, -2), mode=mode)
    (finite_difference_of_object,) = finite_difference_op(linear_gradient_object)

    # Verify correct values excluding borders
    torch.testing.assert_close(finite_difference_of_object[0, 0, 1:-1], (1 - 1j) * torch.ones(18))
    torch.testing.assert_close(finite_difference_of_object[1, 1:-1, 0], (2 - 2j) * torch.ones(18))


@pytest.mark.parametrize('pad_mode', ['zeros', 'circular'])
@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (-4,), (1, 3)])
def test_finite_difference_op_adjointness(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> None:
    """Test finite difference operator adjoint property."""
    dotproduct_adjointness_test(*create_finite_difference_op_and_range_domain(dim, mode, pad_mode))


@pytest.mark.parametrize('pad_mode', ['zeros', 'circular'])
@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (-4,), (1, 3)])
def test_finite_difference_op_grad(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> None:
    """Test the gradient of finite difference operator."""
    gradient_of_linear_operator_test(*create_finite_difference_op_and_range_domain(dim, mode, pad_mode))


@pytest.mark.parametrize('pad_mode', ['zeros', 'circular'])
@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (-4,), (1, 3)])
def test_finite_difference_op_forward_mode_autodiff(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> None:
    """Test the forward-mode autodiff of the finite difference operator."""
    forward_mode_autodiff_of_linear_operator_test(*create_finite_difference_op_and_range_domain(dim, mode, pad_mode))
