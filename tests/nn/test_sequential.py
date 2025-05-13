"""Tests for Sequential module."""

import pytest
from mrpro.nn import FiLM, Sequential
from mrpro.operators import FastFourierOp
from mrpro.utils import RandomGenerator
from torch.nn import Linear


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('input_shape', 'emb_shape'),
    [
        ((1, 32), (1, 16)),
        ((2, 64), None),
    ],
)
def test_sequential(input_shape, emb_shape, device):
    """Test Sequential output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    emb = rng.float32_tensor(emb_shape).to(device).requires_grad_(True) if emb_shape else None
    seq = Sequential(
        Linear(input_shape[1], 64),
        FastFourierOp(),
        FiLM(channels=64, channels_emb=16),
    ).to(device)
    output = seq(x, emb)
    assert output.shape == (input_shape[0], 32), f'Output shape {output.shape} != expected {(input_shape[0], 32)}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.isnan().any(), 'NaN values in input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert seq[0].weight.grad is not None, 'No gradient computed for first Linear'
    assert seq[2].weight.grad is not None, 'No gradient computed for second Linear'
