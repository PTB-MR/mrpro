"""Tests for Phase Operator."""

import pytest
import torch
from mrpro.operators import PhaseOp

from tests import RandomGenerator
from tests.helper import autodiff_of_operator_test


def test_phase_operator_forward():
    """Test that PhaseOp returns angle of tensors."""
    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((2, 3))
    b = random_generator.complex64_tensor((3, 10))
    phase_op = PhaseOp()
    phase_a, phase_b = phase_op(a, b)
    assert torch.allclose(phase_a, torch.angle(a))
    assert torch.allclose(phase_b, torch.angle(b))


@pytest.mark.filterwarnings('ignore:Anomaly Detection has been enabled')
def test_autodiff_magnitude_operator():
    """Test autodiff works for magnitude operator."""
    random_generator = RandomGenerator(seed=2)
    a = random_generator.complex64_tensor((5, 9, 8))
    b = random_generator.complex64_tensor((10, 11, 12))
    autodiff_of_operator_test(PhaseOp(), a, b)
