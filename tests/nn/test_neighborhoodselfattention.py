"""Tests for NeighborhoodSelfAttention module."""

import pytest
from mrpro.nn import NeighborhoodSelfAttention
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_channels_in', 'n_channels_out', 'n_heads', 'kernel_size', 'input_shape', 'features_last'),
    [
        (2, 3, 1, 2, (1, 2, 16, 16), False),
        (3, 2, 2, 4, (1, 3, 8, 8, 8, 8), True),
    ],
)
def test_neighborhood_self_attention(
    n_channels_in: int,
    n_channels_out: int,
    n_heads: int,
    kernel_size: int,
    input_shape: tuple[int, ...],
    features_last: bool,
    device: str,
) -> None:
    """Test NeighborhoodSelfAttention output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)

    if features_last:
        x = x.moveaxis(1, -1)

    attn = NeighborhoodSelfAttention(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        n_heads=n_heads,
        kernel_size=kernel_size,
        features_last=features_last,
    ).to(device)

    output = attn(x)

    expected_shape = (x.shape[0], n_channels_out, *x.shape[2:])
    assert output.shape == expected_shape
    assert not output.isnan().any(), 'NaN values in output'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'

    assert attn.to_qkv.weight.grad is not None, 'No gradient computed for to_qkv.weight'
    assert attn.to_qkv.bias.grad is not None, 'No gradient computed for to_qkv.bias'
    assert attn.to_out.weight.grad is not None, 'No gradient computed for to_out.weight'
    assert attn.to_out.bias.grad is not None, 'No gradient computed for to_out.bias'


@pytest.mark.parametrize(
    ('kernel_size', 'dilation', 'circular'),
    [
        (3, 1, False),
        (5, 2, True),
        (7, 1, False),
    ],
)
def test_neighborhood_attention_variants(kernel_size: int, dilation: int, circular: bool) -> None:
    """Test NeighborhoodSelfAttention with different neighborhood configurations."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 32, 16, 16)).requires_grad_(True)

    attn = NeighborhoodSelfAttention(
        n_channels_in=32,
        n_channels_out=32,
        n_heads=4,
        kernel_size=kernel_size,
        dilation=dilation,
        circular=circular,
    )

    output = attn(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
