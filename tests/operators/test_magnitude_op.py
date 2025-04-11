"""Tests for Magnitude Operator."""

import torch
from mrpro.operators import MagnitudeOp
from mrpro.utils import RandomGenerator

from tests import autodiff_test


def test_magnitude_operator_forward():
    """Test that MagnitudeOp returns abs of tensors."""
    rng = RandomGenerator(seed=2)
    a = rng.complex64_tensor((2, 3))
    b = rng.complex64_tensor((3, 10))
    magnitude_op = MagnitudeOp()
    magnitude_a, magnitude_b = magnitude_op(a, b)
    assert torch.allclose(magnitude_a, torch.abs(a))
    assert torch.allclose(magnitude_b, torch.abs(b))


def test_autodiff_magnitude_operator():
    """Test autodiff works for magnitude operator."""
    rng = RandomGenerator(seed=2)
    a = rng.complex64_tensor((5, 9, 8))
    b = rng.complex64_tensor((10, 11, 12))
    autodiff_test(MagnitudeOp(), a, b)
