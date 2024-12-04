"""Tests for saturation recovery signal model."""

import pytest
import torch
from mrpro.operators.models import SaturationRecovery
from tests import autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '0'),  # short ti
        (20, 'm0'),  # long ti
    ],
)
def test_saturation_recovery(ti, result):
    """Test for saturation recovery.

    Checking that idata output tensor at ti=0 is close to 0. Checking
    that idata output tensor at large ti is close to m0.
    """
    model = SaturationRecovery(ti)
    m0, t1 = create_parameter_tensor_tuples()
    (image,) = model(m0, t1)

    zeros = torch.zeros_like(m0)

    # Assert closeness to zero for ti=0
    if result == '0':
        torch.testing.assert_close(image[0, ...], zeros)
    # Assert closeness to m0 for large ti
    elif result == 'm0':
        torch.testing.assert_close(image[0, ...], m0)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_saturation_recovery_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (ti,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = SaturationRecovery(ti)
    m0, t1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=2)
    (signal,) = model_op(m0, t1)
    assert signal.shape == signal_shape


def test_autodiff_aturation_recovery():
    """Test autodiff works for aturation recovery model."""
    model = SaturationRecovery(ti=10)
    m0, t1 = create_parameter_tensor_tuples((2, 5, 10, 10, 10), number_of_tensors=2)
    autodiff_test(model, m0, t1)
