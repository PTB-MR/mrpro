"""Tests for FiLM module."""

import pytest
from mrpro.nn.FiLM import FiLM
from mrpro.utils.RandomGenerator import RandomGenerator


@pytest.mark.parametrize(
    ('channels', 'channels_emb', 'input_shape', 'emb_shape'),
    [
        (64, 32, (1, 64, 32, 32), (1, 32)),
        (32, 16, (2, 32, 16, 16), (2, 16)),
    ],
)
def test_film(channels, channels_emb, input_shape, emb_shape):
    """Test FiLM output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).requires_grad_(True)
    emb = rng.float32_tensor(emb_shape).requires_grad_(True)
    film = FiLM(channels=channels, channels_emb=channels_emb)
    output = film(x, emb)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert emb.grad is not None, 'No gradient computed for embedding'
    assert not x.isnan().any(), 'NaN values in input'
    assert not emb.isnan().any(), 'NaN values in embedding'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert not emb.grad.isnan().any(), 'NaN values in embedding gradients'
    assert film.project[1].weight.grad is not None, 'No gradient computed for Linear layer'
