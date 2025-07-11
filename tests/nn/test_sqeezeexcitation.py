"""Tests for SqueezeExcitation module."""

import pytest
from mrpro.nn import SqueezeExcitation
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    ('dim', 'input_shape', 'squeeze_channels'),
    [
        (2, (1, 64, 32, 32), 16),
        (3, (1, 64, 16, 16, 16), 16),
    ],
)
def test_squeeze_excitation(dim, input_shape, squeeze_channels):
    """Test SqueezeExcitation output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).requires_grad_(True)
    se = SqueezeExcitation(dim=dim, input_channels=input_shape[1], squeeze_channels=squeeze_channels)
    output = se(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.isnan().any(), 'NaN values in input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert se.scale[1].weight.grad is not None, 'No gradient computed for Conv'
