"""Tests for MOLLI signal model."""

import pytest
import torch
from mrpro.operators.models import MOLLI
from tests import autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS, create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '-ac'),  # short ti
        (1e8, 'a'),  # long ti
    ],
)
def test_molli(ti, result):
    """Test for MOLLI.

    Checking that idata output tensor at ti=0 is close to a. Checking
    that idata output tensor at large ti is close to a(1-c).
    """
    a, t1, c = create_parameter_tensor_tuples(number_of_tensors=3)

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    (image,) = model(a, c, t1)

    # Assert closeness to a(1-c) for large ti
    if result == '-ac':
        torch.testing.assert_close(image[0, ...], -a * c)
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


@pytest.mark.cuda
def test_molli_cuda():
    """Test the molli model works on cuda devices."""
    a, b, t1 = create_parameter_tensor_tuples((2, 5, 10, 10, 10), number_of_tensors=3)

    # Create on CPU, transfer to GPU and run on GPU
    model = MOLLI(ti=10)
    model.cuda()
    (signal,) = model(a.cuda(), b.cuda(), t1.cuda())
    assert signal.is_cuda

    # Create on GPU and run on GPU
    model = MOLLI(ti=torch.as_tensor(10).cuda())
    (signal,) = model(a.cuda(), b.cuda(), t1.cuda())
    assert signal.is_cuda

    # Create on GPU, transfer to CPU and run on CPU
    model = MOLLI(ti=torch.as_tensor(10).cuda())
    model.cpu()
    (signal,) = model(a, b, t1)
    assert signal.is_cpu
