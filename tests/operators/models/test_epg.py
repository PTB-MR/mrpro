"""Tests for EPG signal models."""

from collections.abc import Sequence

import torch
from mrpro.operators.models import CardiacFingerprinting
from tests import RandomGenerator


def test_cmrf_model(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test the CMRF model."""
    acquisition_times = torch.linspace(0, 10, 705)
    model = CardiacFingerprinting(acquisition_times=acquisition_times, echo_time=0.05)
    rng = RandomGenerator(0)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    signal = model(t1, t2, m0)
    assert signal
