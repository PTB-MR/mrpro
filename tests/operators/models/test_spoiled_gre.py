"""Tests for spoiled gradient echo signal model."""

from collections.abc import Sequence

import pytest
import torch
from mr2.operators.models import SpoiledGRE
from mr2.utils import RandomGenerator
from tests import autodiff_test
from tests.operators.models.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS


def test_spoiled_gre_special_values(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test spoiled gradient echo signal at special input values."""
    rng = RandomGenerator(1)
    m0 = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=2)
    t2star = rng.float32_tensor(parameter_shape, low=1e-10, high=0.5)

    # flip angle = 0 -> no signal
    (actual,) = SpoiledGRE(flip_angle=0.0, echo_time=1.0, repetition_time=1.0)(m0, t1, t2star)
    assert torch.isclose(actual, torch.tensor(0.0j)).all()

    # repetition time = 0 -> no signal
    (actual,) = SpoiledGRE(flip_angle=1.0, echo_time=1.0, repetition_time=0.0)(m0, t1, t2star)
    assert torch.isclose(actual, torch.tensor(0.0j)).all()

    # echo time = 0 -> signal independent of t2star
    constant_m0 = m0.mean(None, keepdim=True)
    constant_t1 = t1.mean(None, keepdim=True)
    (actual,) = SpoiledGRE(flip_angle=1.0, echo_time=0.0, repetition_time=1.0)(constant_m0, constant_t1, t2star)
    assert torch.isclose(actual, actual.ravel()[0]).all()


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_spoiled_gre_shape(
    parameter_shape: Sequence[int], contrast_dim_shape: Sequence[int], signal_shape: Sequence[int]
) -> None:
    """Test correct signal shapes."""
    rng = RandomGenerator(1)
    m0 = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=2)
    t2star = rng.float32_tensor(parameter_shape, low=1e-10, high=0.5)
    echo_time = rng.float32_tensor(contrast_dim_shape, low=1e-10, high=0.1)
    repetition_time = rng.float32_tensor(contrast_dim_shape, low=1e-10, high=0.1)
    flip_angle = rng.float32_tensor(contrast_dim_shape, low=0.01, high=0.2)
    model = SpoiledGRE(flip_angle=flip_angle, echo_time=echo_time, repetition_time=repetition_time)
    (signal,) = model(m0, t1, t2star)
    assert signal.shape == signal_shape
    assert signal.isfinite().all()


def test_autodiff_spoiled_gre(parameter_shape: Sequence[int] = (2, 5, 10, 10)) -> None:
    """Test autodiff works for spoiled gre model."""
    model = SpoiledGRE(flip_angle=0.1, echo_time=1e-3, repetition_time=10e-3)
    rng = RandomGenerator(2)
    m0 = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=2)
    t2star = rng.float32_tensor(parameter_shape, low=1e-10, high=0.5)

    autodiff_test(model, m0, t1, t2star)


@pytest.mark.cuda
def test_spoiled_gre_cuda(parameter_shape: Sequence[int] = (2, 5, 10, 10, 10)) -> None:
    """Test the spoiled gre model works on cuda devices."""
    rng = RandomGenerator(1)
    m0 = rng.complex64_tensor(parameter_shape, low=1e-10, high=10)
    t1 = rng.float32_tensor(parameter_shape, low=1e-10, high=2)
    t2star = rng.float32_tensor(parameter_shape, low=1e-10, high=0.5)

    # Create on CPU with tensor parameters, move to GPU and apply on GPU
    model = SpoiledGRE(
        flip_angle=torch.tensor([0.1, 0.1]),
        echo_time=torch.tensor([1e-3, 1e-3]),
        repetition_time=torch.tensor([10e-3, 10e-3]),
    )
    model.cuda()
    (signal,) = model(m0.cuda(), t1.cuda(), t2star.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU and run on GPU
    model = SpoiledGRE(
        flip_angle=torch.tensor([0.1, 0.1]).cuda(),
        echo_time=torch.tensor([1e-3, 1e-3]).cuda(),
        repetition_time=torch.tensor([10e-3, 10e-3]).cuda(),
    )
    (signal,) = model(m0.cuda(), t1.cuda(), t2star.cuda())
    assert signal.is_cuda
    assert signal.isfinite().all()

    # Create on GPU, transfer to CPU and run on CPU
    model = SpoiledGRE(
        flip_angle=torch.tensor([0.1, 0.1]).cuda(),
        echo_time=torch.tensor([1e-3, 1e-3]).cuda(),
        repetition_time=torch.tensor([10e-3, 10e-3]).cuda(),
    )
    model.cpu()
    (signal,) = model(m0.cpu(), t1.cpu(), t2star.cpu())
    assert signal.is_cpu
    assert signal.isfinite().all()
