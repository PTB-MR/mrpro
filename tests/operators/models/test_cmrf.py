from collections.abc import Sequence

import pytest
import torch
from mr2.operators.models.cMRF import CardiacFingerprinting
from mr2.utils import RandomGenerator
from mr2.utils.reshape import unsqueeze_right
from tests import autodiff_test


def test_cmrf_basic(parameter_shape: Sequence[int] = (2, 1, 1, 4, 2)) -> None:
    """Test the CMRF model."""
    acquisition_times = unsqueeze_right(torch.linspace(0, 15, 705), len(parameter_shape)).expand(-1, *parameter_shape)
    model = CardiacFingerprinting(acquisition_times=acquisition_times)
    rng = RandomGenerator(0)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    (signal,) = model(m0, t1, t2)
    assert signal.isfinite().all()
    assert signal.shape == (705, *parameter_shape)


def test_cmrf_autodiff(parameter_shape: Sequence[int] = (2, 1, 1, 4, 2)) -> None:
    """Test autodiff works for cMRF model."""
    rng = RandomGenerator(8)
    acquisition_times = rng.float32_tensor(15, low=0.8, high=2).cumsum(0)
    model = CardiacFingerprinting(acquisition_times=acquisition_times)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    autodiff_test(model, m0, t1, t2)


@pytest.mark.cuda
def test_cmrf_cuda(parameter_shape: Sequence[int] = (2,)) -> None:
    """Test the cMRF model works on cuda devices."""
    rng = RandomGenerator(8)
    acquisition_times = rng.float32_tensor(15, low=0.8, high=2).cumsum(0)

    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)

    # Create on CPU, transfer to GPU and run on GPU
    model = CardiacFingerprinting(acquisition_times)
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = CardiacFingerprinting(acquisition_times.cuda())
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = CardiacFingerprinting(acquisition_times.cuda())
    model.cpu()
    (signal,) = model(m0, t1, t2)
    assert signal.is_cpu
    assert signal.isfinite().all()


def test_cmrf_invalid():
    """Test invalid parameters for cMRF model."""
    with pytest.raises(ValueError, match='acquisition times'):
        CardiacFingerprinting(acquisition_times=torch.ones(16))
    with pytest.raises(ValueError, match='would start before the previous block finished'):
        CardiacFingerprinting(acquisition_times=torch.ones(15))
    with pytest.raises(ValueError, match='would start before the previous block finished'):
        CardiacFingerprinting(repetition_time=5.0)
    with pytest.raises(ValueError, match='should be smaller than repetition time'):
        CardiacFingerprinting(echo_time=0.006)
    with pytest.raises(ValueError, match='Negative echo time'):
        CardiacFingerprinting(t2_prep_echo_times=(-0.01, 0.05, 0.08))
