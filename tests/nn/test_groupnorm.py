"""Tests for GroupNorm module."""

from collections.abc import Sequence

import pytest
from mr2.nn import GroupNorm
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_channels', 'n_groups', 'input_shape', 'affine'),
    [
        (32, None, (1, 32, 32, 32), True),
        (64, 8, (2, 64, 16, 16, 16), False),
    ],
)
def test_groupnorm(
    n_channels: int,
    n_groups: int | None,
    input_shape: Sequence[int],
    device: str,
    affine: bool,
) -> None:
    """Test GroupNorm output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    norm = GroupNorm(n_channels=n_channels, n_groups=n_groups, affine=affine).to(device)
    output = norm(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    if affine:
        assert norm.weight is not None, 'Weight should not be None when affine is True'
        assert norm.weight.grad is not None, 'No gradient computed for weight'
        assert norm.bias is not None, 'Bias should not be None when affine is True'
        assert norm.bias.grad is not None, 'No gradient computed for bias'
