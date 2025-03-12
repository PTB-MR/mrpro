"""Tests for saturation recovery signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import SaturationRecovery
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


@pytest.mark.parametrize(
    ('saturation_time', 'result'),
    [
        (0, '0'),  # short saturation_time
        (60, 'm0'),  # long saturation_time
    ],
)
def test_saturation_recovery_special_values(
    saturation_time: float, result: str, parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)
) -> None:
    """Test saturation recovery signal at special input values."""
    model = SaturationRecovery(saturation_time)
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)

    (signal,) = model(m0, t1)

    # Assert closeness to zero for saturation_time=0
    if result == '0':
        torch.testing.assert_close(signal[0], torch.zeros_like(m0))
    # Assert closeness to m0 for large saturation_time
    elif result == 'm0':
        torch.testing.assert_close(signal[0], m0)
    else:
        raise ValueError(f'Unknown result {result}')


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_saturation_recovery_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(1)
    saturation_time = rng.float32_tensor(contrast_dim_shape, low=-0.1, high=2)
    model = SaturationRecovery(saturation_time)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)
    (signal,) = model(m0, t1)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_aturation_recovery(
    parameter_shape: Sequence[int] = (2, 5, 10),
    contrast_dim_shape=(13, 2, 5, 10),
) -> None:
    """Test autodiff works for aturation recovery model."""
    rng = RandomGenerator(2)
    saturation_time = rng.float32_tensor(contrast_dim_shape, low=-0.1, high=2)
    model = SaturationRecovery(saturation_time)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)
    model = SaturationRecovery(saturation_time=10)
    autodiff_test(model, m0, t1)


@pytest.mark.cuda
def test_saturation_recovery_cuda(
    parameter_shape: Sequence[int] = (2, 5), contrast_dim_shape: Sequence[int] = (13, 2, 5)
) -> None:
    """Test the saturation recovery model works on cuda devices."""
    rng = RandomGenerator(3)
    saturation_time = rng.float32_tensor(contrast_dim_shape, low=-0.1, high=2)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)

    # Create on CPU, transfer to GPU and run on GPU
    model = SaturationRecovery(saturation_time.tolist())
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = SaturationRecovery(saturation_time=saturation_time.cuda())
    (signal,) = model(m0.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = SaturationRecovery(saturation_time=saturation_time.cuda())
    model.cpu()
    (signal,) = model(m0, t1)
    assert signal.is_cpu
    assert signal.isfinite().all()
