"""Tests for MOLLI signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import MOLLI
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '-ac'),  # short ti
        (1e8, 'a'),  # long ti
    ],
)
def test_molli_special_values(ti: float, result: str, parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test MOLLI signal at special input values."""
    rng = RandomGenerator(1)
    a = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    c = rng.float32_tensor(parameter_shape, low=1e-10, high=10)

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    (image,) = model(a, c, t1)

    # Assert closeness to a(1-c) for large ti
    if result == '-ac':
        torch.testing.assert_close(image[0, ...], -a * c)
    # Assert closeness to a for ti=0
    elif result == 'a':
        torch.testing.assert_close(image[0, ...], a)
    else:
        raise ValueError(f'Unknown result {result}')


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_molli_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(1)
    a = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    c = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    ti = rng.float32_tensor(contrast_dim_shape, low=1e-10, high=10)
    model = MOLLI(ti)
    (signal,) = model(a, c, t1)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_molli(parameter_shape: Sequence[int] = (2, 5, 10, 10)) -> None:
    """Test autodiff works for molli model."""
    model = MOLLI(ti=10)
    rng = RandomGenerator(2)
    a = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    c = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    autodiff_test(model, a, c, t1)


@pytest.mark.cuda
def test_molli_cuda(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test the molli model works on cuda devices."""
    rng = RandomGenerator(1)
    a = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    c = rng.float32_tensor(parameter_shape, low=1e-10, high=10)

    # Create on CPU, transfer to GPU and run on GPU
    model = MOLLI(ti=[5, 10])
    model.cuda()
    (signal,) = model(a.cuda(), c.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = MOLLI(ti=torch.tensor((5, 10)).cuda())
    (signal,) = model(a.cuda(), c.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = MOLLI(ti=torch.tensor((5, 10)).cuda())
    model.cpu()
    (signal,) = model(a, c, t1)
    assert signal.is_cpu
    assert signal.isfinite().all()
