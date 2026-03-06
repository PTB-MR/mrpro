"""Tests for MultiEchoSpinEcho signal model."""

from collections.abc import Sequence

import pytest
from mrpro.operators.models.NeuroMRF import DELICS_FLIP_ANGLES, NeuroMRF
from mrpro.utils import RandomGenerator
from tests import autodiff_test


def test_neuro_mrf_basic(parameter_shape: Sequence[int] = (2, 1, 3, 4, 2)) -> None:
    """Test the NeuroMRF model."""
    model = NeuroMRF()
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    relative_b1 = None
    (signal,) = model(m0, t1, t2, relative_b1)
    assert signal.isfinite().all()
    assert signal.shape == (len(DELICS_FLIP_ANGLES), *parameter_shape)


def test_neuro_mrf_autodiff(parameter_shape: Sequence[int] = (2, 1, 3, 4, 2)) -> None:
    """Test autodiff works for NeuroMRF model."""
    rng = RandomGenerator(8)
    model = NeuroMRF()
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.9, high=1.1)
    autodiff_test(model, m0, t1, t2, relative_b1)


@pytest.mark.cuda
def test_mese_cuda(parameter_shape: Sequence[int] = (2,)) -> None:
    """Test the NeuroMRF model works on cuda devices."""
    rng = RandomGenerator(8)

    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    # Create on CPU, transfer to GPU and run on GPU
    model = NeuroMRF()
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = NeuroMRF(DELICS_FLIP_ANGLES.cuda())
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = NeuroMRF(DELICS_FLIP_ANGLES.cuda())
    model.cpu()
    (signal,) = model(m0, t1, t2)
    assert signal.is_cpu
    assert signal.isfinite().all()
