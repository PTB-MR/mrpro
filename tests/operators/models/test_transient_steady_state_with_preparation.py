"""Tests for transient steady state signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import TransientSteadyStateWithPreparation
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


@pytest.mark.parametrize(
    ('sampling_time', 'm0_scaling_preparation', 'result'),
    [
        (0, 1, 'm0'),  # short sampling time without preparation pulse
        (0, 0, '0'),  # short sampling time after saturation pulse
        (0, -1, '-m0'),  # short sampling time after inversion pulse
        (60, 1, 'm0*'),  # long sampling time without preparation pulse
        (60, 0, 'm0*'),  # long sampling time after saturation pulse
        (60, -1, 'm0*'),  # long sampling time after inversion pulse
    ],
)
def test_transient_steady_state_special_values(
    sampling_time: float, m0_scaling_preparation: float, result: str, parameter_shape: Sequence[int] = (2, 5, 10)
) -> None:
    """Test transient steady state signal at special input values."""
    repetition_time = 5
    model = TransientSteadyStateWithPreparation(sampling_time, repetition_time, m0_scaling_preparation)
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    flip_angle = rng.float32_tensor(parameter_shape, low=0.01, high=0.49 * torch.pi)

    (signal,) = model(m0, t1, flip_angle)

    # Assert closeness to m0
    if result == 'm0':
        torch.testing.assert_close(signal[0, ...], m0)
    # Assert closeness to 0
    elif result == '0':
        torch.testing.assert_close(signal[0, ...], torch.zeros_like(m0))
    # Assert closeness to -m0
    elif result == '-m0':
        torch.testing.assert_close(signal[0, ...], -m0)
    # Assert closensess to m0*
    elif result == 'm0*':
        t1_star = 1 / (1 / t1 - torch.log(torch.cos(flip_angle)) / repetition_time)
        m0_star = m0 * t1_star / t1
        torch.testing.assert_close(signal[0, ...], m0_star)
    else:
        raise ValueError(f'Unknown result {result}')


def test_transient_steady_state_inversion_recovery() -> None:
    """Transient steady state as inversion recovery.

    For very small flip angles and long repetition times, the transient steady state should be the same as a
    inversion recovery model.
    """
    t1 = torch.tensor([100, 200, 300, 400, 500, 1000, 2000, 4000])
    flip_angle = torch.full_like(t1, 0.0001)
    m0 = torch.ones_like(t1)
    sampling_time = torch.tensor([0, 100, 400, 800, 2000]).unsqueeze(-1)

    analytical_signal = m0 * (1 - 2 * torch.exp(-(sampling_time / t1)))

    model = TransientSteadyStateWithPreparation(sampling_time, repetition_time=100, m0_scaling_preparation=-1)
    (signal,) = model(m0, t1, flip_angle)

    torch.testing.assert_close(signal, analytical_signal)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_transient_steady_state_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(1)
    sampling_time = rng.float32_tensor(contrast_dim_shape, low=0, high=2)
    if len(parameter_shape) == 1:
        repetition_time: float | torch.Tensor = 5
        m0_scaling_preparation: float | torch.Tensor = 1
        delay_after_preparation: float | torch.Tensor = 0.01
    else:
        repetition_time = rng.float32_tensor(contrast_dim_shape[1:], low=0, high=2)
        m0_scaling_preparation = rng.complex64_tensor(contrast_dim_shape[1:])
        delay_after_preparation = rng.float32_tensor(contrast_dim_shape[1:], low=0, high=1)
    model = TransientSteadyStateWithPreparation(
        sampling_time, repetition_time, m0_scaling_preparation, delay_after_preparation
    )
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)
    flip_angle = rng.float32_tensor(parameter_shape, low=0.01, high=0.49 * torch.pi)
    (signal,) = model(m0, t1, flip_angle)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_transient_steady_state(
    parameter_shape: Sequence[int] = (2, 5, 10), contrast_dim_shape: Sequence[int] = (6,)
) -> None:
    """Test autodiff works for transient steady state model."""
    rng = RandomGenerator(2)
    sampling_time = rng.float32_tensor(contrast_dim_shape, low=0, high=2)
    repetition_time = rng.float32_tensor(parameter_shape, low=0, high=2)
    m0_scaling_preparation = rng.complex64_tensor(parameter_shape)
    delay_after_preparation = rng.float32_tensor(parameter_shape, low=0, high=1)
    model = TransientSteadyStateWithPreparation(
        sampling_time, repetition_time, m0_scaling_preparation, delay_after_preparation
    )
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)
    flip_angle = rng.float32_tensor(parameter_shape, low=0.01, high=0.49 * torch.pi)
    autodiff_test(model, m0, t1, flip_angle)


@pytest.mark.cuda
def test_transient_steady_state_cuda(
    parameter_shape: Sequence[int] = (2, 5, 10), contrast_dim_shape: Sequence[int] = (6,)
) -> None:
    """Test the transient steady state model works on cuda devices."""
    rng = RandomGenerator(3)

    sampling_time = rng.float32_tensor(contrast_dim_shape, low=0, high=2)
    repetition_time, delay_after_preparation = rng.float32_tensor((2, *parameter_shape), low=0, high=2)
    m0_scaling_preparation = rng.complex64_tensor(parameter_shape)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)
    flip_angle = rng.float32_tensor(parameter_shape, low=0.01, high=0.49 * torch.pi)

    # Create on CPU, transfer to GPU and run on GPU
    model = TransientSteadyStateWithPreparation(
        sampling_time.tolist(), repetition_time, m0_scaling_preparation, delay_after_preparation
    )
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), flip_angle.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = TransientSteadyStateWithPreparation(
        sampling_time.cuda(), repetition_time, m0_scaling_preparation, delay_after_preparation
    )
    (signal,) = model(m0.cuda(), t1.cuda(), flip_angle.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = TransientSteadyStateWithPreparation(
        sampling_time.cuda(), repetition_time, m0_scaling_preparation, delay_after_preparation
    )
    model.cpu()
    (signal,) = model(m0, t1, flip_angle)
    assert signal.is_cpu
    assert signal.isfinite().all()
