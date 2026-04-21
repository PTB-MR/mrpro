"""Tests for symmetrized gradient operator."""

from collections.abc import Sequence
from typing import Literal

import pytest
import torch
from einops import repeat
from mrpro.operators import SymmetrizedGradientOp
from mrpro.utils import RandomGenerator

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_symmetrized_gradient_op_and_range_domain(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> tuple[SymmetrizedGradientOp, torch.Tensor, torch.Tensor]:
    """Create a symmetrized gradient operator and an element from domain and range."""
    rng = RandomGenerator(seed=0)
    input_shape = (len(dim), 6, 4, 10, 20, 16)  # First dimension matches number of gradients

    # Generate symmetrized gradient operator
    symmetrized_gradient_op = SymmetrizedGradientOp(dim, mode, pad_mode)

    u = rng.complex64_tensor(size=input_shape)
    v = rng.complex64_tensor(size=(len(dim), *input_shape))
    return symmetrized_gradient_op, u, v


@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
def test_symmetrized_gradient_op_forward(
    mode: Literal['central', 'forward', 'backward'],
) -> None:
    """Test symmetrized gradient of a simple linear vector field in 2D."""
    # Create a test object v = (v0, v1) in 2D
    size = 10
    y_coords = repeat(torch.arange(size, dtype=torch.float32), 'y -> 1 y x', x=size)
    x_coords = repeat(torch.arange(size, dtype=torch.float32), 'x -> 1 y x', y=size)
    v0 = x_coords + 2 * y_coords  # v0(y, x) = x + 2y
    v1 = 2 * x_coords + y_coords  # v1(y, x) = 2x + y
    v0 = v0 - 1j * v0
    v1 = v1 - 1j * v1
    v = torch.cat((v0, v1), dim=0)  # shape (2, size, size)

    # Generate and apply symmetrized gradient operator
    dim = (-1, -2)
    assert v.shape[0] == len(dim)  # Ensure first dimension of v matches number of dimensions in dim
    sym_grad_op = SymmetrizedGradientOp(dim=dim, mode=mode)
    (sym_grad,) = sym_grad_op(v)  # shape (2, 2, size, size)

    # Extract interior (remove borders to avoid boundary effects)
    sym_00 = sym_grad[0, 0, 1:-1, 1:-1]
    sym_11 = sym_grad[1, 1, 1:-1, 1:-1]
    sym_01 = sym_grad[0, 1, 1:-1, 1:-1]
    sym_10 = sym_grad[1, 0, 1:-1, 1:-1]

    # Verify correct values excluding borders
    torch.testing.assert_close(sym_00, (1 - 1j) * torch.ones_like(sym_00))
    torch.testing.assert_close(sym_11, (1 - 1j) * torch.ones_like(sym_11))
    torch.testing.assert_close(sym_01, (2 - 2j) * torch.ones_like(sym_01))
    torch.testing.assert_close(sym_10, (2 - 2j) * torch.ones_like(sym_10))


@pytest.mark.parametrize('pad_mode', ['zeros', 'circular'])
@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (-4,), (1, 3)])
def test_symmetrized_gradient_op_adjointness(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> None:
    """Test symmetrized gradient operator adjoint property."""
    dotproduct_adjointness_test(*create_symmetrized_gradient_op_and_range_domain(dim, mode, pad_mode))


@pytest.mark.parametrize('pad_mode', ['zeros', 'circular'])
@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (-4,), (1, 3)])
def test_symmetrized_gradient_op_grad(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> None:
    """Test the gradient of symmetrized gradient operator."""
    gradient_of_linear_operator_test(*create_symmetrized_gradient_op_and_range_domain(dim, mode, pad_mode))


@pytest.mark.parametrize('pad_mode', ['zeros', 'circular'])
@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (-4,), (1, 3)])
def test_symmetrized_gradient_op_forward_mode_autodiff(
    dim: Sequence[int], mode: Literal['central', 'forward', 'backward'], pad_mode: Literal['zeros', 'circular']
) -> None:
    """Test the forward-mode autodiff of the symmetrized gradient operator."""
    forward_mode_autodiff_of_linear_operator_test(*create_symmetrized_gradient_op_and_range_domain(dim, mode, pad_mode))


@pytest.mark.cuda
def test_symmetrized_gradient_op_cuda() -> None:
    """Test symmetrized gradient operator works on CUDA devices."""

    # Set dimensional parameters
    dim = (-3, -2, -1)
    input_shape = (len(dim), 6, 4, 10, 20, 16)

    # Generate data
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=input_shape)

    # Create on CPU, run on CPU
    symmetrized_gradient_op = SymmetrizedGradientOp(dim, mode='central', pad_mode='circular')
    operator = symmetrized_gradient_op.H @ symmetrized_gradient_op
    (symmetrized_gradient_output,) = operator(u)
    assert symmetrized_gradient_output.is_cpu

    # Transfer to GPU, run on GPU
    symmetrized_gradient_op = SymmetrizedGradientOp(dim, mode='central', pad_mode='circular')
    operator = symmetrized_gradient_op.H @ symmetrized_gradient_op
    operator.cuda()
    (symmetrized_gradient_output,) = operator(u.cuda())
    assert symmetrized_gradient_output.is_cuda
