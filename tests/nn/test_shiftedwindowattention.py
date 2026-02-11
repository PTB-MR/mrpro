"""Tests for ShiftedWindowAttention module."""

import pytest
from mr2.nn.attention import ShiftedWindowAttention
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_dim', 'window_size', 'shifted'),
    [
        (2, 8, False),
        (4, 4, True),
    ],
)
def test_shifted_window_attention(n_dim: int, window_size: int, shifted: bool, device: str) -> None:
    """Test ShiftedWindowAttention output shape and backpropagation."""
    n_batch, n_channels, n_heads = 2, 8, 2
    spatial_shape = (window_size * 4,) * n_dim
    rng = RandomGenerator(13)
    x = rng.float32_tensor((n_batch, n_channels, *spatial_shape)).to(device).requires_grad_(True)
    swin = ShiftedWindowAttention(
        n_dim=n_dim,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
        n_heads=n_heads,
        window_size=window_size,
        shifted=shifted,
    ).to(device)
    out = swin(x)
    assert out.shape == x.shape, f'Output shape {out.shape} != input shape {x.shape}'
    assert not out.isnan().any(), 'NaN values in output'
    out.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert swin.to_qkv.weight.grad is not None, 'No gradient computed for to_qkv.weight'
    assert swin.relative_position_bias_table.grad is not None, 'No gradient computed for relative_position_bias_table'


@pytest.mark.parametrize('shifted', [True, False], ids=['shifted', 'non-shifted'])
def test_shifted_window_attention_size_mismatch(shifted: bool):
    n_batch, n_channels, n_heads, n_dim, window_size = 3, 4, 2, 2, 7
    spatial_shape = (window_size * 4 + 1,) * n_dim
    rng = RandomGenerator(13)
    x = rng.float32_tensor((n_batch, n_channels, *spatial_shape))
    swin = ShiftedWindowAttention(
        n_dim=n_dim,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
        n_heads=n_heads,
        window_size=window_size,
        shifted=shifted,
    )
    out = swin(x)
    assert out.shape == x.shape, f'Output shape {out.shape} != input shape {x.shape}'
