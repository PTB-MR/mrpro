"""Tests for the WASABITI signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import WASABITI
from mrpro.utils.reshape import unsqueeze_right
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


def test_WASABITI_shift(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test symmetry property of shifted and unshifted WASABITI spectra."""
    rng = RandomGenerator(1)

    offsets = torch.linspace(-300, 300, 13)
    recovery_time_symmetric = rng.float32_tensor((13, *parameter_shape), low=0.01, high=2)
    recovery_time_symmetric = recovery_time_symmetric + recovery_time_symmetric.flip(0)  # symmetric recovery time
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    t1 = rng.float32_tensor(parameter_shape)
    no_b0_shift = torch.zeros(parameter_shape)

    # with no B0 shift and symmetric offsets and symmetric recover time,
    # the signal should be symmetric around the center
    model = WASABITI(offsets=offsets, recovery_time=recovery_time_symmetric)
    (signal,) = model(no_b0_shift, relative_b1, t1)
    torch.testing.assert_close(signal, torch.flip(signal, dims=(0,)))
    assert signal.isfinite().all()

    # with shifted offsets the signal should be symmetric around a different center
    offsets_shifted = offsets + 100
    recovery_time_constant = rng.float32_tensor((1, *parameter_shape), low=0.01, high=2)  # constant recovery time
    model = WASABITI(offsets=offsets_shifted, recovery_time=recovery_time_constant)
    (signal_offsetsshifted,) = model(no_b0_shift, relative_b1, t1)
    lower_index = int((offsets_shifted == -200).nonzero())
    upper_index = int((offsets_shifted == 200).nonzero())
    torch.testing.assert_close(signal_offsetsshifted[lower_index], signal_offsetsshifted[upper_index])
    assert signal_offsetsshifted.isfinite().all()

    # if both offsets and b0 are shifted in the same way, we should get the same signal
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    offsets_b0_shifted = unsqueeze_right(offsets, b0_shift.ndim) + b0_shift
    model = WASABITI(offsets=offsets_b0_shifted, recovery_time=recovery_time_symmetric)
    (signal_matching_shift,) = model(b0_shift, relative_b1, t1)
    torch.testing.assert_close(signal, signal_matching_shift)


def test_WASABITI_relaxation_term(
    parameter_shape: Sequence[int] = (2, 5), contrast_dim_shape: Sequence[int] = (13, 2, 5)
) -> None:
    """Test relaxation term (Mzi) of WASABITI model."""

    offsets = torch.full(contrast_dim_shape, 50000)  # Far off-resonance
    b0_shift = torch.zeros(parameter_shape)
    relative_b1 = torch.ones(parameter_shape)
    rng = RandomGenerator(2)

    recovery_time = rng.float32_tensor(contrast_dim_shape, low=0.01, high=2)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)

    model = WASABITI(offsets=offsets, recovery_time=recovery_time)
    (signal,) = model(b0_shift, relative_b1, t1)
    excepted = 1 - torch.exp(-recovery_time / t1)
    torch.testing.assert_close(signal, excepted)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_WASABITI_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(3)
    offsets = rng.float32_tensor(contrast_dim_shape, low=-300, high=300)
    recovery_time = rng.float32_tensor(contrast_dim_shape, low=0.01, high=2)
    model = WASABITI(offsets=offsets, recovery_time=recovery_time)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    t1 = rng.float32_tensor(parameter_shape, low=0.001, high=2)
    (signal,) = model(b0_shift, relative_b1, t1)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_WASABITI(
    parameter_shape: Sequence[int] = (2, 5, 10), contrast_dim_shape: Sequence[int] = (13, 2, 5, 10)
) -> None:
    """Test autodiff works for WASABITI model."""
    rng = RandomGenerator(4)
    offsets = rng.float32_tensor(contrast_dim_shape, low=-300, high=300)
    recovery_time = rng.float32_tensor(contrast_dim_shape, low=-0.1, high=10)
    model = WASABITI(offsets=offsets, recovery_time=recovery_time)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    t1 = rng.float32_tensor(parameter_shape, low=0.0, high=5)
    autodiff_test(model, b0_shift, relative_b1, t1)


@pytest.mark.cuda
def test_wasabiti_cuda(parameter_shape: Sequence[int] = (2, 5), contrast_dim_shape: Sequence[int] = (13, 2, 5)) -> None:
    """Test the WASABITI model works on cuda devices."""
    rng = RandomGenerator(5)
    offsets = rng.float32_tensor(contrast_dim_shape, low=-300, high=300)
    recovery_time = rng.float32_tensor(contrast_dim_shape, low=0.01, high=2)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    t1 = rng.float32_tensor(parameter_shape, low=0.01, high=2)

    # Create on CPU, transfer to GPU and run on GPU
    model = WASABITI(offsets=offsets, recovery_time=recovery_time)
    model.cuda()
    (signal,) = model(b0_shift.cuda(), relative_b1.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = WASABITI(offsets=offsets.cuda(), recovery_time=recovery_time)
    (signal,) = model(b0_shift.cuda(), relative_b1.cuda(), t1.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = WASABITI(offsets=offsets.cuda(), recovery_time=recovery_time)
    model.cpu()
    (signal,) = model(b0_shift, relative_b1, t1)
    assert signal.is_cpu
    assert signal.isfinite().all()
