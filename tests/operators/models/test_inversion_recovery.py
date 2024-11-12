"""Tests for inversion recovery signal model."""

import pytest
import torch
from mrpro.operators.models import InversionRecovery
from tests import autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '-m0'),  # short ti
        (20, 'm0'),  # long ti
    ],
)
def test_inversion_recovery(ti, result):
    """Test for inversion recovery.

    Checking that idata output tensor at ti=0 is close to -m0. Checking
    that idata output tensor at large ti is close to m0.
    """
    model = InversionRecovery(ti)
    m0, t1 = create_parameter_tensor_tuples()
    (image,) = model(m0, t1)

    # Assert closeness to -m0 for ti=0
    if result == '-m0':
        torch.testing.assert_close(image[0, ...], -m0)
    # Assert closeness to m0 for large ti
    elif result == 'm0':
        torch.testing.assert_close(image[0, ...], m0)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_inversion_recovery_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (ti,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = InversionRecovery(ti)
    m0, t1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=2)
    (signal,) = model_op(m0, t1)
    assert signal.shape == signal_shape


def test_autodiff_inversion_recovery():
    """Test autodiff works for inversion_recovery model."""
    model = InversionRecovery(ti=10)
    m0, t1 = create_parameter_tensor_tuples(parameter_shape=(2, 5, 10, 10, 10), number_of_tensors=2)
    autodiff_test(model, m0, t1)
