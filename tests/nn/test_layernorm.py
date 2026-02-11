"""Tests for LayerNorm module."""

from collections.abc import Sequence

import pytest
import torch
from mr2.nn.LayerNorm import LayerNorm
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
def test_layernorm_basic(
    n_channels: int | None,
    features_last: bool,
    input_shape: Sequence[int],
    device: str,
) -> None:
    """Test LayerNorm basic functionality."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    norm = LayerNorm(n_channels=n_channels, features_last=features_last).to(device)
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


@pytest.mark.parametrize(
    ('n_channels', 'cond_dim', 'input_shape', 'cond_shape'),
    [
        (32, 16, (1, 32, 32, 32), (1, 16)),
        (64, 32, (2, 64, 16, 16), (2, 32)),
    ],
)
def test_layernorm_with_conditioning(
    n_channels: int,
    cond_dim: int,
    input_shape: Sequence[int],
    cond_shape: Sequence[int],
) -> None:
    """Test LayerNorm with conditioning."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).requires_grad_(True)
    cond = rng.float32_tensor(cond_shape).requires_grad_(True)
    norm = LayerNorm(n_channels=n_channels, cond_dim=cond_dim)

    output = norm(x, cond=cond)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert cond.grad is not None, 'No gradient computed for conditioning'
    assert norm.cond_proj is not None, 'cond_proj should not be None when cond_dim > 0'
    assert norm.cond_proj.weight.grad is not None, 'No gradient computed for cond_proj'


def test_layernorm_features_last() -> None:
    """Test LayerNorm with features_last=True vs features_last=False."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).requires_grad_(True)

    norm_standard = LayerNorm(n_channels=3, features_last=False)
    y_standard = norm_standard(x)

    norm_last = LayerNorm(n_channels=3, features_last=True)
    y_last = norm_last(x.moveaxis(1, -1))

    torch.testing.assert_close(y_standard, y_last.moveaxis(-1, 1))


def test_layernorm_no_channels() -> None:
    """Test LayerNorm without channels (pure normalization)."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 32, 32, 32)).requires_grad_(True)
    norm = LayerNorm(n_channels=None)

    output = norm(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'

    # Check that normalization is applied over channel dim (dim=1 for features_last=False)
    mean = output.mean(dim=1, keepdim=True)
    var = (output * output).mean(dim=1, keepdim=True) - mean * mean

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), 'Mean not close to 0'
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3), 'Variance not close to 1'


def test_layernorm_conditioning_without_channels() -> None:
    """Test LayerNorm with conditioning but no channels (should raise error)."""
    with pytest.raises(ValueError, match='channels must be provided if cond_dim > 0'):
        LayerNorm(n_channels=None, cond_dim=16)


def test_layernorm_invalid_cond_dim() -> None:
    """Test LayerNorm with invalid cond_dim."""
    with pytest.raises(RuntimeError, match='Trying to create tensor with negative dimension'):
        LayerNorm(n_channels=32, cond_dim=-1)


def test_layernorm_3d_input() -> None:
    """Test LayerNorm with 3D input."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((2, 64, 128)).requires_grad_(True)
    norm = LayerNorm(n_channels=64)

    output = norm(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'


def test_layernorm_5d_input() -> None:
    """Test LayerNorm with 5D input."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 32, 16, 16, 16)).requires_grad_(True)
    norm = LayerNorm(n_channels=32)

    output = norm(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'


def test_layernorm_conditioning_features_last() -> None:
    """Test LayerNorm with conditioning and features_last=True."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).requires_grad_(True)
    cond = rng.float32_tensor((1, 8)).requires_grad_(True)

    norm = LayerNorm(n_channels=3, features_last=True, cond_dim=8)
    output = norm(x.moveaxis(1, -1), cond=cond)

    assert output.shape == x.moveaxis(1, -1).shape, f'Output shape {output.shape} != expected shape'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert cond.grad is not None, 'No gradient computed for conditioning'


def test_layernorm_gradient_flow() -> None:
    """Test that gradients flow properly through LayerNorm."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 32, 32, 32)).requires_grad_(True)
    norm = LayerNorm(n_channels=32)

    output = norm(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients are computed for all learnable parameters
    assert x.grad is not None, 'Input gradients not computed'
    assert norm.weight is not None, 'Weight should not be None when n_channels is provided'
    assert norm.bias is not None, 'Bias should not be None when n_channels is provided'
    assert norm.weight.grad is not None, 'Weight gradients not computed'
    assert norm.bias.grad is not None, 'Bias gradients not computed'

    # Check that gradients are finite
    assert torch.isfinite(x.grad).all(), 'Input gradients contain non-finite values'
    assert torch.isfinite(norm.weight.grad).all(), 'Weight gradients contain non-finite values'
    assert torch.isfinite(norm.bias.grad).all(), 'Bias gradients contain non-finite values'
