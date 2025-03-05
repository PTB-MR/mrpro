"""Tests for the WASABI signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models import WASABI
from mrpro.utils.reshape import unsqueeze_right
from tests import RandomGenerator, autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


def test_WASABI_shift(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)):
    """Test symmetry property of shifted and unshifted WASABI spectra."""
    rng = RandomGenerator(1)

    offsets = torch.linspace(-300, 300, 13)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    c = rng.complex64_tensor(parameter_shape)
    d = rng.complex64_tensor(parameter_shape)
    no_b0_shift = torch.zeros(parameter_shape)

    # with no B0 shift and symmetric offsets the signal should be symmetric around the center
    model = WASABI(offsets=offsets)
    (signal,) = model(no_b0_shift, relative_b1, c, d)
    torch.testing.assert_close(signal, torch.flip(signal, dims=(0,)))
    assert signal.isfinite().all()

    # with asymmetric offsets the signal should be symmetric around a different center
    offsets_shifted = offsets + 100
    model = WASABI(offsets=offsets_shifted)
    (signal_offsetsshifted,) = model(no_b0_shift, relative_b1, c, d)
    lower_index = int((offsets_shifted == -200).nonzero())
    upper_index = int((offsets_shifted == 200).nonzero())
    torch.testing.assert_close(signal_offsetsshifted[lower_index], signal_offsetsshifted[upper_index])
    assert signal_offsetsshifted.isfinite().all()

    # if both offsets and b0 are shifted in the same way, we should get the same signal
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    offsets_b0_shifted = unsqueeze_right(offsets, b0_shift.ndim) + b0_shift
    model = WASABI(offsets=offsets_b0_shifted)
    (signal_matching_shift,) = model(b0_shift, relative_b1, c, d)
    torch.testing.assert_close(signal, signal_matching_shift)


def test_WASABI_scaling(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test signal scaling property of WASABI model."""
    rng = RandomGenerator(37)
    offsets = rng.float32_tensor(13, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    c = rng.complex64_tensor(parameter_shape)
    d = rng.complex64_tensor(parameter_shape)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    model = WASABI(offsets=offsets)
    (signal,) = model(b0_shift, relative_b1, c, d)
    assert signal.isfinite().all()

    # WASABI should be linearly scaling with c and d
    scale = rng.complex64_tensor(parameter_shape)
    (signal_scaled,) = model(b0_shift, relative_b1, c * scale, d * scale)
    torch.testing.assert_close(scale * signal, signal_scaled)


def test_WASABI_extreme_offset(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test signal for extreme offset."""
    rng = RandomGenerator(37)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    c = rng.complex64_tensor(parameter_shape)
    d = rng.complex64_tensor(parameter_shape)
    offset = [500000]
    model = WASABI(offsets=offset)
    (signal,) = model(b0_shift, relative_b1, c, d)
    # For an extreme offset, the signal should be unattenuated
    torch.testing.assert_close(signal[0], c)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_WASABI_shape(parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]):
    """Test correct signal shapes."""
    rng = RandomGenerator(8)
    offsets = rng.float32_tensor(contrast_dim_shape, low=-100, high=100)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    c = rng.complex64_tensor(parameter_shape)
    d = rng.complex64_tensor(parameter_shape)
    model = WASABI(offsets)
    (signal,) = model(b0_shift, relative_b1, c, d)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_WASABI(parameter_shape: Sequence[int] = (2, 5, 3), contrast_dim_shape: Sequence[int] = (3,)) -> None:
    """Test autodiff works for WASABI model."""
    rng = RandomGenerator(8)
    offsets = rng.float32_tensor(contrast_dim_shape, low=-100, high=100)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    c = rng.complex64_tensor(parameter_shape)
    d = rng.complex64_tensor(parameter_shape)
    model = WASABI(offsets=offsets)
    autodiff_test(model, b0_shift, relative_b1, c, d)


@pytest.mark.cuda
def test_wasabi_cuda(parameter_shape: Sequence[int] = (2, 5, 3), contrast_dim_shape: Sequence[int] = (3,)) -> None:
    """Test the WASABI model works on cuda devices."""
    rng = RandomGenerator(8)
    offset = rng.float32_tensor(contrast_dim_shape, low=-100, high=100)
    b0_shift = rng.float32_tensor(parameter_shape, low=-100, high=100)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.1, high=2)
    c = rng.complex64_tensor(parameter_shape)
    d = rng.complex64_tensor(parameter_shape)

    # Create on CPU, transfer to GPU and run on GPU
    model = WASABI(offset)
    model.cuda()
    (signal,) = model(b0_shift.cuda(), relative_b1.cuda(), c.cuda(), d.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = WASABI(offset.cuda())
    (signal,) = model(b0_shift.cuda(), relative_b1.cuda(), c.cuda(), d.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = WASABI(offset.cuda())
    model.cpu()
    (signal,) = model(b0_shift, relative_b1, c, d)
    assert signal.is_cpu
    assert signal.isfinite().all()
