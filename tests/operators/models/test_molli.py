"""Tests for MOLLI signal model."""

import pytest
import torch
from mrpro.operators.models import MOLLI
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, 'a-b'),  # short ti
        (20, 'a'),  # long ti
    ],
)
def test_molli(ti, result):
    """Test for MOLLI.

    Checking that idata output tensor at ti=0 is close to a. Checking
    that idata output tensor at large ti is close to a-b.
    """
    # Generate qdata tensor, not random as a<b is necessary for t1_star to be >= 0
    other, coils, z, y, x = 10, 5, 100, 100, 100
    a = torch.ones((other, coils, z, y, x)) * 2
    b = torch.ones((other, coils, z, y, x)) * 5
    t1 = torch.ones((other, coils, z, y, x)) * 2

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    (image,) = model.forward(a, b, t1)

    # Assert closeness to a-b for large ti
    if result == 'a-b':
        torch.testing.assert_close(image[0, ...], a - b)
    # Assert closeness to a for ti=0
    elif result == 'a':
        torch.testing.assert_close(image[0, ...], a)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_molli_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (ti,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = MOLLI(ti)
    a, b, t1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=3)
    (signal,) = model_op.forward(a, b, t1)
    assert signal.shape == signal_shape
