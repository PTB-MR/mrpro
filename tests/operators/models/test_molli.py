"""Tests for MOLLI signal model."""

import pytest
import torch
from mrpro.operators.models import MOLLI
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, 'a(1-c)'),  # short ti
        (20, 'a'),  # long ti
    ],
)
def test_molli(ti, result):
    """Test for MOLLI.

    Checking that idata output tensor at ti=0 is close to a. Checking
    that idata output tensor at large ti is close to a(1-c).
    """
    a, t1 = create_parameter_tensor_tuples()
    # c>2 is necessary for t1_star to be >= 0 and to ensure t1_star < t1
    random_generator = RandomGenerator(seed=0)
    c = random_generator.float32_tensor(size=a.shape, low=2.0, high=4.0)

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    (image,) = model(a, c, t1)

    # Assert closeness to a(1-c) for large ti
    if result == 'a(1-c)':
        torch.testing.assert_close(image[0, ...], a * (1 - c))
    # Assert closeness to a for ti=0
    elif result == 'a':
        torch.testing.assert_close(image[0, ...], a)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_molli_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (ti,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = MOLLI(ti)
    a, c, t1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=3)
    (signal,) = model_op(a, c, t1)
    assert signal.shape == signal_shape


def test_autodiff_molli():
    """Test autodiff works for molli model."""
    model = MOLLI(ti=10)
    a, b, t1 = create_parameter_tensor_tuples((2, 5, 10, 10, 10), number_of_tensors=3)
    autodiff_test(model, a, b, t1)
