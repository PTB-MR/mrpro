"""Tests for Phase Operator."""

import pytest
import torch
from mrpro.operators import PhaseOp

from tests import RandomGenerator, autodiff_test


def test_phase_operator_forward():
    """Test that PhaseOp returns angle of tensors."""
    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((2, 3))
    b = random_generator.complex64_tensor((3, 10))
    phase_op = PhaseOp()
    phase_a, phase_b = phase_op(a, b)
    assert torch.allclose(phase_a, torch.angle(a))
    assert torch.allclose(phase_b, torch.angle(b))


def test_autodiff_magnitude_operator():
    """Test autodiff works for magnitude operator."""
    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((5, 9, 8))
    b = random_generator.complex64_tensor((10, 11, 12))
    autodiff_test(PhaseOp(), a, b)


@pytest.mark.cuda
def test_phase_operator_cuda():
    """Test that PhaseOp works on CUDA devices."""
    # Generate random tensors
    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((2, 3))
    b = random_generator.complex64_tensor((3, 10))

    # Create on CPU, run on CPU
    phase_op = PhaseOp()
    phase_a, phase_b = phase_op(a, b)
    assert phase_a.is_cpu
    assert phase_b.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    phase_op = PhaseOp()
    phase_op.cuda()
    phase_a, phase_b = phase_op(a.cuda(), b.cuda())
    assert phase_a.is_cuda
    assert phase_b.is_cuda
