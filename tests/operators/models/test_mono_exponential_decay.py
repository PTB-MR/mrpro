"""Tests for the mono-exponential decay signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import MonoExponentialDecay
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


@pytest.mark.parametrize(
    ('decay_time', 'result'),
    [
        (0, 'm0'),  # short decay time
        (60, '0'),  # long decay time
    ],
)
def test_mono_exponential_decay_special_values(
    decay_time: float, result: str, parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)
) -> None:
    """Test for mono-exponential decay signal model.

    Check that idata output tensor at ti=0 is close to 0.
    Check that idata output tensor at large ti is close to m0.
    """
    rng = RandomGenerator(1)
    m0 = rng.complex64_tensor(parameter_shape)
    decay_constant = rng.float32_tensor(parameter_shape, low=1e-3, high=2)
    model = MonoExponentialDecay(decay_time)
    (signal,) = model(m0, decay_constant)

    zeros = torch.zeros_like(m0)

    # Assert closeness to m0 for short decay_time
    if result == '0':
        torch.testing.assert_close(signal[0, ...], zeros)
    # Assert closeness to 0 for large decay_time
    elif result == 'm0':
        torch.testing.assert_close(signal[0, ...], m0)
    else:
        raise ValueError(f'Unknown result {result}')


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_mono_exponential_decay_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(1)
    decay_time = rng.float32_tensor(contrast_dim_shape, low=-0.1, high=2)
    m0 = rng.complex64_tensor(parameter_shape)
    decay_constant = rng.float32_tensor(parameter_shape, low=0.001, high=2)
    model = MonoExponentialDecay(decay_time)
    (signal,) = model(m0, decay_constant)
    assert signal.shape == signal_shape
    assert torch.isfinite(signal).all()


def test_autodiff_mono_exponential_decay(parameter_shape: Sequence[int] = (2, 5, 2, 10, 10)) -> None:
    """Test autodiff works for mono-exponential decay model."""
    model = MonoExponentialDecay(decay_time=(-0.1, 0, 1, 5))
    rng = RandomGenerator(3)
    m0 = rng.complex64_tensor(parameter_shape)
    decay_constant = rng.float32_tensor(parameter_shape, low=0.01, high=5)
    autodiff_test(model, m0, decay_constant)


@pytest.mark.cuda
def test_mono_exponential_decay_cuda(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test the mono-exponential decay model works on cuda devices."""
    rng = RandomGenerator(3)
    m0 = rng.complex64_tensor(parameter_shape)
    decay_constant = rng.float32_tensor(parameter_shape, low=0.01, high=5)
    # Create on CPU, transfer to GPU and run on GPU
    model = MonoExponentialDecay(decay_time=(-0.1, 0, 1, 5))
    model.cuda()
    (signal,) = model(m0.cuda(), decay_constant.cuda())
    assert signal.is_cuda
    assert torch.isfinite(signal).all()

    # Create on GPU and run on GPU
    model = MonoExponentialDecay(decay_time=torch.tensor((-0.1, 0, 1, 5)).cuda())
    (signal,) = model(m0.cuda(), decay_constant.cuda())
    assert signal.is_cuda
    assert torch.isfinite(signal).all()

    # Create on GPU, transfer to CPU and run on CPU
    model = MonoExponentialDecay(decay_time=torch.tensor((-0.1, 0, 1, 5)).cuda())
    model.cpu()
    (signal,) = model(m0, decay_constant)
    assert signal.is_cpu
    assert torch.isfinite(signal).all()
