import pytest
import torch
from mrpro.nn import ShiftedWindowAttention


@pytest.mark.parametrize(
    'dim,window_size,shifted',
    [
        (2, 4, False),
        (2, 4, True),
        (3, 2, False),
        (3, 2, True),
    ],
)
def test_shifted_window_attention_forward_and_grad(dim: int, window_size: int   , shifted)->:
    batch = 2
    channels = 8
    n_heads = 2
    spatial_shape = (window_size * 2,) * dim
    x = torch.randn((batch, channels) + spatial_shape, requires_grad=True)

    attn = ShiftedWindowAttention(dim=dim, channels=channels, n_heads=n_heads, window_size=window_size, shifted=shifted)

    out = attn(x)
    assert out.shape == x.shape, f'Output shape {out.shape} != input shape {x.shape}'

    # Check backward
    out.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
