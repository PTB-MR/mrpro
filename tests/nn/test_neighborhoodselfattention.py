"""Tests for NeighborhoodSelfAttention module."""

import pytest
import torch
from mrpro.nn.attention.NeighborhoodSelfAttention import NeighborhoodSelfAttention
from mrpro.utils import RandomGenerator
from tests.conftest import minimal_torch_26


@minimal_torch_26
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
    ids=['2d_kernel2', '4d_features-last_kernel4'],
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

    attn = NeighborhoodSelfAttention(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        n_heads=n_heads,
        kernel_size=kernel_size,
        features_last=features_last,
    ).to(device)

    if features_last:
        output = attn(x.moveaxis(1, -1)).moveaxis(-1, 1)
    else:
        output = attn(x)

    expected_shape = (input_shape[0], n_channels_out, *input_shape[2:])
    assert output.shape == expected_shape
    assert not output.isnan().any(), 'NaN values in output'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'

    assert attn.to_qkv.weight.grad is not None, 'No gradient computed for to_qkv.weight'
    assert attn.to_qkv.bias.grad is not None, 'No gradient computed for to_qkv.bias'
    assert attn.to_out.weight.grad is not None, 'No gradient computed for to_out.weight'
    assert attn.to_out.bias.grad is not None, 'No gradient computed for to_out.bias'


@minimal_torch_26
@pytest.mark.parametrize(
    ('kernel_size', 'dilation', 'circular', 'rope'),
    [
        (3, 1, False, True),
        (5, 2, True, False),
        (7, 1, False, True),
    ],
)
def test_neighborhood_attention_variants(kernel_size: int, dilation: int, circular: bool, rope: bool) -> None:
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
        rope_embed_fraction=1.0 if rope else 0.0,
    )

    output = attn(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'


@pytest.mark.parametrize(
    ('kernel_size', 'circular', 'input_shape'),
    [
        (11, False, (1, 8, 32, 32)),
        (7, True, (1, 8, 16, 16)),
    ],
    ids=['regular', 'circular'],
)
def test_neighborhood_constraint(kernel_size: int, circular: bool, input_shape: tuple[int, int, int, int]) -> None:
    """Test that neighborhood attention only affects pixels within the kernel window."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).requires_grad_(True)

    attn = NeighborhoodSelfAttention(
        n_channels_in=8,
        n_channels_out=8,
        n_heads=2,
        kernel_size=kernel_size,
        dilation=1,
        circular=circular,
    )

    output_original = attn(x)
    x_modified = x.clone()
    test_point = (input_shape[-2] - 2, input_shape[-1] - 2)
    x_modified[..., test_point[0], test_point[1]] += 1.0
    output_modified = attn(x_modified)

    diff = output_modified - output_original
    changed_pixels = torch.abs(diff).sum(dim=(0, 1)) > 1e-6

    half_kernel = kernel_size // 2
    h, w = input_shape[2], input_shape[3]

    i_coords, j_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    if circular:
        h_dist = torch.minimum((i_coords - test_point[0]) % h, (test_point[0] - i_coords) % h)
        w_dist = torch.minimum((j_coords - test_point[1]) % w, (test_point[1] - j_coords) % w)
        in_neighborhood = (h_dist <= half_kernel) & (w_dist <= half_kernel)
    else:
        h_min, h_max = max(0, test_point[0] - half_kernel), min(h, test_point[0] + half_kernel + 1)
        w_min, w_max = max(0, test_point[1] - half_kernel), min(w, test_point[1] + half_kernel + 1)
        in_neighborhood = (i_coords >= h_min) & (i_coords < h_max) & (j_coords >= w_min) & (j_coords < w_max)

    neighborhood_changed = changed_pixels[in_neighborhood].all()
    outside_changed = changed_pixels[~in_neighborhood].any()

    assert neighborhood_changed, 'Not all pixels in the neighborhood changed, which indicates a problem'
    assert not outside_changed, 'Pixels outside the neighborhood changed, which violates the constraint'
