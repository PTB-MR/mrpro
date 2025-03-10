"""Tests for Magnitude Operator."""

import pytest
import torch
from mrpro.operators import MagnitudeOp

from tests import RandomGenerator, autodiff_test


def test_magnitude_operator_forward():
    """Test that MagnitudeOp returns abs of tensors."""
    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((2, 3))
    b = random_generator.complex64_tensor((3, 10))
    magnitude_op = MagnitudeOp()
    magnitude_a, magnitude_b = magnitude_op(a, b)
    assert torch.allclose(magnitude_a, torch.abs(a))
    assert torch.allclose(magnitude_b, torch.abs(b))


def test_autodiff_magnitude_operator():
    """Test autodiff works for magnitude operator."""
    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((5, 9, 8))
    b = random_generator.complex64_tensor((10, 11, 12))
    autodiff_test(MagnitudeOp(), a, b)


@pytest.mark.cuda
def test_magnitude_op_cuda() -> None:
    """Test magnitude operator works on CUDA devices."""

    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((2, 3))
    b = random_generator.complex64_tensor((3, 10))

    # Create on CPU, run on CPU
    magnitude_op = MagnitudeOp()
    magnitude_a, magnitude_b = magnitude_op(a, b)
    assert magnitude_a.is_cpu
    assert magnitude_b.is_cpu

    # Transfer to GPU, run on GPU
    magnitude_op = MagnitudeOp()
    magnitude_op.cuda()
    magnitude_a, magnitude_b = magnitude_op(a.cuda(), b.cuda())
    assert magnitude_a.is_cuda
    assert magnitude_b.is_cuda
