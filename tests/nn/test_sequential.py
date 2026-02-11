"""Tests for Sequential module."""

from collections.abc import Sequence

import pytest
from mr2.nn import FiLM, Sequential
from mr2.operators import FastFourierOp, MagnitudeOp
from mr2.utils import RandomGenerator
from torch.nn import Linear


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('input_shape', 'cond_dim'),
    [
        ((1, 32), (1, 16)),
        ((2, 32), None),
    ],
)
def test_sequential(
    input_shape: Sequence[int],
    cond_dim: Sequence[int] | None,
    device: str,
) -> None:
    """Test Sequential output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    cond = rng.float32_tensor(cond_dim).to(device).requires_grad_(True) if cond_dim else None
    seq = Sequential(
        Linear(input_shape[1], 64),
        FastFourierOp(dim=(-1,)),
        FiLM(channels=64, cond_dim=16),
        MagnitudeOp(),
    ).to(device)
    output = seq(x, cond=cond)
    assert output.shape == (input_shape[0], 64)
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    if cond is not None:
        assert cond.grad is not None, 'No gradient computed for cond'
        assert not cond.grad.isnan().any(), 'NaN values in cond gradients'
    assert seq[0].weight.grad is not None, 'No gradient computed for Linear'
