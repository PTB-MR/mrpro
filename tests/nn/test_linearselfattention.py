"""Tests for LinearSelfAttention module."""

import pytest
from mr2.nn.attention import LinearSelfAttention
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_channels_in', 'n_channels_out', 'n_heads', 'input_shape', 'features_last'),
    [
        (32, 32, 4, (1, 32, 32, 32), False),
        (64, 64, 8, (2, 64, 16, 16), False),
        (16, 16, 2, (1, 16, 16, 16), True),
    ],
)
def test_linear_self_attention(
    n_channels_in: int,
    n_channels_out: int,
    n_heads: int,
    input_shape: tuple[int, ...],
    features_last: bool,
    device: str,
) -> None:
    """Test LinearSelfAttention output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)

    attn = LinearSelfAttention(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        n_heads=n_heads,
        features_last=features_last,
    ).to(device)

    if features_last:
        output = attn(x.moveaxis(1, -1)).moveaxis(-1, 1)
    else:
        output = attn(x)

    expected_shape = (x.shape[0], n_channels_out, *x.shape[2:])
    assert output.shape == expected_shape, f'Output shape {output.shape} != expected shape {expected_shape}'
    assert not output.isnan().any(), 'NaN values in output'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'

    assert attn.to_qkv.weight.grad is not None, 'No gradient computed for to_qkv.weight'
    assert attn.to_qkv.bias.grad is not None, 'No gradient computed for to_qkv.bias'
    assert attn.to_out.weight.grad is not None, 'No gradient computed for to_out.weight'
    assert attn.to_out.bias.grad is not None, 'No gradient computed for to_out.bias'
