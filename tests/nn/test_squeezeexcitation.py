"""Tests for SqueezeExcitation module."""

from collections.abc import Sequence

import pytest
from mr2.nn.attention import SqueezeExcitation
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    ('dim', 'input_shape', 'squeeze_channels'),
    [
        (2, (1, 64, 32, 32), 16),
        (3, (1, 64, 16, 16, 16), 16),
    ],
)
def test_squeeze_excitation(
    dim: int,
    input_shape: Sequence[int],
    squeeze_channels: int,
) -> None:
    """Test SqueezeExcitation output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).requires_grad_(True)
    se = SqueezeExcitation(n_dim=dim, n_channels_input=input_shape[1], n_channels_squeeze=squeeze_channels)
    output = se(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert se.scale[1].weight.grad is not None, 'No gradient computed for Conv'
