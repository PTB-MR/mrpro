"""Tests for PEX signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import PEX
from mrpro.utils import RandomGenerator
from tests import autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


@pytest.mark.parametrize(
    ('voltages', 'pulse_duration', 'expected_behavior'),
    [
        (0, 1.0, 'unity'),  # zero voltage should give signal close to 1
        ([0, 100, 1000], 0, 'unity'),  # zero pulse duration should give signal close to 1
    ],
)
def test_pex_special_values(
    voltages: float | list[float],
    pulse_duration: float,
    expected_behavior: str,
    parameter_shape: Sequence[int] = (2, 5, 10, 10, 10),
) -> None:
    """Test PEX signal at special input values."""
    rng = RandomGenerator(0)
    prep_delay = 0.01  # short prep delay

    model = PEX(voltages=voltages, prep_delay=prep_delay, pulse_duration=pulse_duration)
    b1 = rng.float32_tensor(parameter_shape, low=0.1, high=10)  # µT/sqrt(kW)
    t1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)

    (signal,) = model(b1, t1)

    # For zero voltage or zero pulse duration, signal should be close to 1
    if expected_behavior == 'unity':
        expected = torch.ones_like(signal)
        torch.testing.assert_close(signal, expected, atol=1e-3, rtol=1e-3)


def test_pex_flip_angle_behavior(parameter_shape: Sequence[int] = (2, 5, 10)) -> None:
    """Test PEX signal behavior with increasing flip angles."""
    rng = RandomGenerator(1)
    voltages = [10, 50, 100]
    prep_delay = 0.01
    pulse_duration = 0.001
    model = PEX(voltages=voltages, prep_delay=prep_delay, pulse_duration=pulse_duration)
    b1 = rng.float32_tensor(parameter_shape, low=1, high=5)  # µT/sqrt(kW)
    t1 = rng.float32_tensor(parameter_shape, low=0.5, high=2)

    (signal,) = model(b1, t1)

    # Signal should decrease with increasing voltage (higher flip angles)
    assert torch.all(signal[0] >= signal[1])  # first voltage < second voltage
    assert torch.all(signal[1] >= signal[2])  # second voltage < third voltage
    assert signal.isfinite().all()


def test_pex_t1_recovery(parameter_shape: Sequence[int] = (2, 5, 10)) -> None:
    """Test PEX signal T1 recovery behavior."""
    rng = RandomGenerator(2)
    voltages = 100  # fixed voltage
    prep_delay = torch.tensor([0.001, 0.01, 0.1, 1.0])  # increasing prep delays
    pulse_duration = 0.001

    model = PEX(voltages=voltages, prep_delay=prep_delay, pulse_duration=pulse_duration)
    b1 = rng.float32_tensor(parameter_shape, low=1, high=5)
    t1 = rng.float32_tensor(parameter_shape, low=0.5, high=2)

    (signal,) = model(b1, t1)

    # Signal should increase with longer prep delay (more T1 recovery)
    for i in range(len(prep_delay) - 1):
        assert torch.all(signal[i] <= signal[i + 1])
    assert signal.isfinite().all()


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_pex_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(1)
    voltages = rng.float32_tensor(contrast_dim_shape, low=0, high=200)
    prep_delay = 0.01
    pulse_duration = 0.001

    model = PEX(voltages=voltages, prep_delay=prep_delay, pulse_duration=pulse_duration)
    b1 = rng.float32_tensor(parameter_shape, low=0.1, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)
    (signal,) = model(b1, t1)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_pex(
    parameter_shape: Sequence[int] = (2, 5, 10),
    contrast_dim_shape: Sequence[int] = (13, 2, 5, 10),
) -> None:
    """Test autodiff works for PEX model."""
    rng = RandomGenerator(2)
    voltages = rng.float32_tensor(contrast_dim_shape, low=0, high=200)
    prep_delay = 0.01
    pulse_duration = 0.001

    model = PEX(voltages=voltages, prep_delay=prep_delay, pulse_duration=pulse_duration)
    b1 = rng.float32_tensor(parameter_shape, low=0.1, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)
    autodiff_test(model, b1, t1)


@pytest.mark.cuda
def test_pex_cuda(parameter_shape: Sequence[int] = (2, 5), contrast_dim_shape: Sequence[int] = (13, 2, 5)) -> None:
    """Test the PEX model works on cuda devices."""
    rng = RandomGenerator(3)
    voltages = rng.float32_tensor(contrast_dim_shape, low=0, high=200)
    prep_delay = 0.01
    pulse_duration = 0.001
    b1 = rng.float32_tensor(parameter_shape, low=0.1, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)

    # Create on CPU, transfer to GPU and run on GPU
    model = PEX(voltages=voltages.tolist(), prep_delay=prep_delay, pulse_duration=pulse_duration)
    model.cuda()
    (signal,) = model(b1.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = PEX(voltages=voltages.cuda(), prep_delay=prep_delay, pulse_duration=pulse_duration)
    (signal,) = model(b1.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = PEX(voltages=voltages.cuda(), prep_delay=prep_delay, pulse_duration=pulse_duration)
    model.cpu()
    (signal,) = model(b1, t1)
    assert signal.is_cpu
    assert signal.isfinite().all()


def test_pex_n_tx_scaling(parameter_shape: Sequence[int] = (2, 5, 10)) -> None:
    """Test PEX signal scales correctly with number of transmit channels."""
    rng = RandomGenerator(4)
    voltages = 100
    prep_delay = 0.01
    pulse_duration = 0.001
    b1 = rng.float32_tensor(parameter_shape, low=1, high=5)
    t1 = rng.float32_tensor(parameter_shape, low=0.5, high=2)

    # Test with different n_tx values
    model_1tx = PEX(voltages=voltages, prep_delay=prep_delay, pulse_duration=pulse_duration, n_tx=1)
    model_4tx = PEX(voltages=voltages, prep_delay=prep_delay, pulse_duration=pulse_duration, n_tx=4)

    (signal_1tx,) = model_1tx(b1, t1)
    (signal_4tx,) = model_4tx(b1, t1)

    # With higher n_tx, the effective voltage is scaled by sqrt(n_tx), so flip angle increases
    # This should result in lower signal values
    assert torch.all(signal_4tx <= signal_1tx)
    assert signal_1tx.isfinite().all()
    assert signal_4tx.isfinite().all()
