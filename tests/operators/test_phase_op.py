"""Tests for Phase Operator."""

import torch
from mrpro.operators import PhaseOp
from mrpro.utils import RandomGenerator

from tests import autodiff_test


def test_phase_operator_forward():
    """Test that PhaseOp returns angle of tensors."""
    rng = RandomGenerator(seed=2)
    a = rng.complex64_tensor((2, 3))
    b = rng.complex64_tensor((3, 10))
    phase_op = PhaseOp()
    phase_a, phase_b = phase_op(a, b)
    assert torch.allclose(phase_a, torch.angle(a))
    assert torch.allclose(phase_b, torch.angle(b))


def test_autodiff_magnitude_operator():
    """Test autodiff works for magnitude operator."""
    rng = RandomGenerator(seed=2)
    a = rng.complex64_tensor((5, 9, 8))
    b = rng.complex64_tensor((10, 11, 12))
    autodiff_test(PhaseOp(), a, b)
