import pytest
from mrpro.nn import ShiftedWindowAttention
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('dim', 'window_size', 'shifted'),
    [
        (2, 8, False),
        (4, 4, True),
    ],
)
def test_shifted_window_attentio(dim: int, window_size: int, shifted: bool, device: str) -> None:
    batch = 2
    channels = 8
    n_heads = 2
    spatial_shape = (window_size * 4,) * dim
    rng = RandomGenerator(13)
    x = rng.float32_tensor((batch, channels, *spatial_shape)).to(device).requires_grad_(True)
    swin = ShiftedWindowAttention(
        dim=dim, channels_in=channels, channels_out=channels, n_heads=n_heads, window_size=window_size, shifted=shifted
    ).to(device)
    out = swin(x)
    assert out.shape == x.shape, f'Output shape {out.shape} != input shape {x.shape}'
    assert not out.isnan().any(), 'NaN values in output'
    out.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert swin.to_qkv.weight.grad is not None, 'No gradient computed for to_qkv.weight'
    assert swin.relative_position_bias_table.grad is not None, 'No gradient computed for relative_position_bias_table'
