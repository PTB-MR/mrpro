"""Tests for GroupNorm32 module."""

import pytest
from mrpro.nn import GroupNorm32
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('channels', 'groups', 'input_shape'),
    [
        (32, None, (1, 32, 32, 32)),
        (64, 8, (2, 64, 16, 16, 16)),
    ],
)
def test_groupnorm32(channels, groups, input_shape, device):
    """Test GroupNorm32 output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    norm = GroupNorm32(channels=channels, groups=groups).to(device)
    output = norm(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.isnan().any(), 'NaN values in input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert norm.weight.grad is not None, 'No gradient computed for weight'
    assert norm.bias.grad is not None, 'No gradient computed for bias'
