"""Tests for RMSNorm module."""

from collections.abc import Sequence

import pytest
import torch
from mr2.nn import RMSNorm
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_channels', 'features_last', 'input_shape'),
    [
        (32, False, (1, 32, 32, 32)),
        (64, True, (2, 16, 16, 64)),
        (None, False, (1, 32, 32, 32)),
        (None, True, (2, 16, 16, 64)),
    ],
)
def test_rmsnorm_basic(n_channels: int | None, features_last: bool, input_shape: Sequence[int], device: str) -> None:
    """Test RMSNorm basic functionality."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    norm = RMSNorm(n_channels=n_channels, features_last=features_last).to(device)
    output = norm(x)

    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'

    if n_channels is not None:
        assert norm.weight is not None, 'Weight should not be None when n_channels is provided'
        assert norm.bias is not None, 'Bias should not be None when n_channels is provided'
        assert norm.weight.grad is not None, 'No gradient computed for weight'
        assert norm.bias.grad is not None, 'No gradient computed for bias'


def test_rmsnorm_features_last() -> None:
    """Test RMSNorm with features_last=True vs features_last=False."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).requires_grad_(True)

    norm_standard = RMSNorm(n_channels=3, features_last=False)
    y_standard = norm_standard(x)

    norm_last = RMSNorm(n_channels=3, features_last=True)
    y_last = norm_last(x.moveaxis(1, -1))

    torch.testing.assert_close(y_standard, y_last.moveaxis(-1, 1))
