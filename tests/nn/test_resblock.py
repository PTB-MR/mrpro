"""Tests for ResBlock module."""

import pytest

from mrpro.nn import ResBlock
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('dim', 'channels_in', 'channels_out', 'channels_emb', 'input_shape', 'emb_shape'),
    [
        (2, 32, 32, 16, (1, 32, 32, 32), (1, 16)),
        (3, 64, 32, 0, (2, 64, 16, 16, 16), None),
    ],
)
def test_resblock(dim, channels_in, channels_out, channels_emb, input_shape, emb_shape, device):
    """Test ResBlock output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    emb = rng.float32_tensor(emb_shape).to(device).requires_grad_(True) if emb_shape else None
    res = ResBlock(dim=dim, channels_in=channels_in, channels_out=channels_out, channels_emb=channels_emb).to(device)
    output = res(x, emb)
    assert output.shape == (input_shape[0], channels_out, *input_shape[2:]), (
        f'Output shape {output.shape} != expected {(input_shape[0], channels_out, *input_shape[2:])}'
    )
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.isnan().any(), 'NaN values in input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert res.block[2].weight.grad is not None, 'No gradient computed for first Conv'
    assert res.block[5].weight.grad is not None, 'No gradient computed for second Conv'
    if emb is not None:
        assert emb.grad is not None, 'No gradient computed for embedding'
        assert not emb.isnan().any(), 'NaN values in embedding'
        assert not emb.grad.isnan().any(), 'NaN values in embedding gradients'
