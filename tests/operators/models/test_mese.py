"""Tests for MultiEchoSpinEcho signal model."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.models.MESE import MultiEchoSpinEcho
from mrpro.utils import RandomGenerator
from tests import autodiff_test


def test_mese_basic(parameter_shape: Sequence[int] = (2, 1, 3, 4, 2)) -> None:
    """Test the MultiEchoSpinEcho model."""
    n_echos = 10
    model = MultiEchoSpinEcho(
        flip_angles=torch.full((n_echos,), torch.pi),
        rf_phases=torch.full((n_echos,), 0),
        echo_time=0.018,
    )
    rng = RandomGenerator(0)
    m0 = rng.complex64_tensor(parameter_shape)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    relative_b1 = None
    (signal,) = model(m0, t1, t2, relative_b1)
    assert signal.isfinite().all()
    assert signal.shape == (n_echos, *parameter_shape)


def test_mese_autodiff(parameter_shape: Sequence[int] = (2, 1, 3, 4, 2)) -> None:
    """Test autodiff works for MultiEchoSpinEcho model."""
    rng = RandomGenerator(8)
    n_echos = 10
    model = MultiEchoSpinEcho(
        flip_angles=torch.full((n_echos,), torch.pi),
        rf_phases=torch.full((n_echos,), 0.0),
    )
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    relative_b1 = rng.float32_tensor(parameter_shape, low=0.9, high=1.1)
    autodiff_test(model, m0, t1, t2, relative_b1)


@pytest.mark.cuda
def test_mese_cuda(parameter_shape: Sequence[int] = (2,)) -> None:
    """Test the MultiEchoSpinEcho model works on cuda devices."""
    rng = RandomGenerator(8)
    n_echos = 10
    flip_angles = rng.float32_tensor(n_echos, low=1e-5, high=5)
    rf_phases = rng.float32_tensor(n_echos, low=1e-5, high=0.5)
    t1 = rng.float32_tensor(parameter_shape, low=1e-5, high=5)
    t2 = rng.float32_tensor(parameter_shape, low=1e-5, high=0.5)
    m0 = rng.complex64_tensor(parameter_shape)
    # Create on CPU, transfer to GPU and run on GPU
    model = MultiEchoSpinEcho(flip_angles=flip_angles, rf_phases=rf_phases)
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = MultiEchoSpinEcho(flip_angles=flip_angles.cuda(), rf_phases=rf_phases.cuda())
    (signal,) = model(m0.cuda(), t1.cuda(), t2.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = MultiEchoSpinEcho(flip_angles=flip_angles.cuda(), rf_phases=rf_phases.cuda())
    model.cpu()
    (signal,) = model(m0, t1, t2)
    assert signal.is_cpu
    assert signal.isfinite().all()
