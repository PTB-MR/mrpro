"""Tests for EPG signal models."""

import pytest
import torch
from mrpro.operators.models.EPG import CardiacFingerprinting, acquisition
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


def test_cmrf_model():
    """Test the CMRF model."""
    acquisition_times = torch.linspace(0, 10, 705)
    te = 0.05
    model = CardiacFingerprinting(acquisition_times=acquisition_times, te=te)
    t1, t2, m0 = create_parameter_tensor_tuples(number_of_tensors=3)
    signal = model(t1, t2, m0)
