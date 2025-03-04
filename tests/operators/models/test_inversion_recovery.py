"""Tests for inversion recovery signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import InversionRecovery
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '-m0'),  # short ti
        (60, 'm0'),  # long ti
    ],
)
def test_inversion_recovery_special_values(
    ti: float, result: str, parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)
) -> None:
    """Test inversion recovery signal at special input values."""
    model = InversionRecovery(ti)
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=1)
    (image,) = model(m0, t1)

    # Assert closeness to -m0 for ti=0
    if result == '-m0':
        torch.testing.assert_close(image[0, ...], -m0)
    # Assert closeness to m0 for large ti
    elif result == 'm0':
        torch.testing.assert_close(image[0, ...], m0)
    else:
        raise ValueError(f'Unknown result {result}')


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_inversion_recovery_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    ti = rng.float32_tensor(contrast_dim_shape, low=1e-10, high=10)
    model = InversionRecovery(ti)
    (signal,) = model(m0, t1)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_inversion_recovery(parameter_shape: Sequence[int] = (2, 5, 10, 10)) -> None:
    """Test autodiff works for inversion_recovery model."""
    model = InversionRecovery(ti=10)
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=10)
    autodiff_test(model, m0, t1)


@pytest.mark.cuda
def test_inversion_recovery_cuda(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test the inversion recovery model works on cuda devices."""
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=10)

    # Create on CPU, transfer to GPU and run on GPU
    model = InversionRecovery(ti=[5, 10])
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = InversionRecovery(ti=torch.tensor((5, 10)).cuda())
    (signal,) = model(m0.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = InversionRecovery(ti=torch.tensor((5, 10)).cuda())
    model.cpu()
    (signal,) = model(m0, t1)
    assert signal.is_cpu
    assert signal.isfinite().all()
