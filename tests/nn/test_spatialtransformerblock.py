"""Test SpatialTransformerBlock"""

from collections.abc import Sequence
from typing import Literal, cast

import pytest
import torch
from mr2.nn.attention import SpatialTransformerBlock
from mr2.utils import RandomGenerator
from tests.conftest import minimal_torch_26


@minimal_torch_26
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    ('channels', 'cond_dim', 'attention_neighborhood', 'features_last', 'norm', 'input_shape'),
    [
        pytest.param(32, 16, None, True, 'group', (16, 16), id='2d-cond-group-last-global'),
        pytest.param(64, 16, 7, False, 'group', (16, 16), marks=minimal_torch_26, id='2d-cond-group-first-NA'),
        pytest.param(64, 16, 5, True, 'group', (16, 16), marks=minimal_torch_26, id='2d-cond-group-last-NA'),
        pytest.param(64, 0, 7, True, 'rms', (16, 8, 16), marks=minimal_torch_26, id='3d-nocond-rms-last-NA'),
    ],
)
def test_spatialtransformerblock_backward(
    channels: int,
    cond_dim: int,
    attention_neighborhood: int | None,
    features_last: bool,
    norm: Literal['group', 'rms'],
    input_shape: Sequence[int],
    device: str,
    torch_compile: bool,
) -> None:
    """Test SpatialTransformerBlock output shape and backpropagation."""
    if device == 'cpu' and attention_neighborhood is not None:
        pytest.skip(
            'CompiledFlex Attention backward not supported on CPU. https://github.com/pytorch/pytorch/issues/148752'
        )
    rng = RandomGenerator(seed=42)

    x = rng.float32_tensor((1, channels, *input_shape)).to(device).requires_grad_(True)
    cond = rng.float32_tensor((1, cond_dim)).to(device).requires_grad_(True) if cond_dim else None

    if features_last:
        dims = tuple(range(-len(input_shape) - 1, -1))
    else:
        dims = tuple(range(-len(input_shape), 0))

    block = SpatialTransformerBlock(
        dim_groups=[dims],
        channels=channels,
        n_heads=4,
        depth=1,
        p_dropout=0,
        cond_dim=cond_dim,
        rope_embed_fraction=0.5,
        attention_neighborhood=attention_neighborhood,
        features_last=features_last,
        norm=norm,
    ).to(device)
    if torch_compile:
        block = cast(SpatialTransformerBlock, torch.compile(block, dynamic=False))
    if features_last:
        output = block(x.moveaxis(1, -1), cond=cond).moveaxis(-1, 1)
    else:
        output = block(x, cond=cond)
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    if cond is not None:
        assert cond.grad is not None, 'No gradient computed for conditioning'
        assert not cond.grad.isnan().any(), 'NaN values in conditioning gradients'


@minimal_torch_26
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize('torch_compile', [False, True])
@pytest.mark.parametrize(
    ('channels', 'cond_dim', 'attention_neighborhood', 'features_last', 'norm', 'input_shape'),
    [
        pytest.param(32, 16, None, True, 'group', (16, 16), id='2d-cond-group-last-global'),
        pytest.param(64, 16, 5, True, 'group', (16, 16), marks=minimal_torch_26, id='2d-cond-group-last-NA'),
        pytest.param(64, 16, 7, False, 'group', (16, 16), marks=minimal_torch_26, id='2d-cond-group-first-NA'),
        pytest.param(64, 0, 7, True, 'rms', (16, 8, 16), marks=minimal_torch_26, id='3d-nocond-rms-last-NA'),
    ],
)
def test_spatialtransformerblock_forward(
    channels: int,
    cond_dim: int,
    attention_neighborhood: int | None,
    features_last: bool,
    norm: Literal['group', 'rms'],
    input_shape: Sequence[int],
    device: str,
    torch_compile: bool,
) -> None:
    """Test SpatialTransformerBlock output shape and backpropagation."""

    rng = RandomGenerator(seed=42)

    x = rng.float32_tensor((1, channels, *input_shape)).to(device).requires_grad_(True)
    cond = rng.float32_tensor((1, cond_dim)).to(device).requires_grad_(True) if cond_dim else None

    if features_last:
        dims = tuple(range(-len(input_shape) - 1, -1))
    else:
        dims = tuple(range(-len(input_shape), 0))

    block = SpatialTransformerBlock(
        dim_groups=[dims],
        channels=channels,
        n_heads=4,
        depth=1,
        p_dropout=0,
        cond_dim=cond_dim,
        rope_embed_fraction=0.5,
        attention_neighborhood=attention_neighborhood,
        features_last=features_last,
        norm=norm,
    ).to(device)
    if torch_compile:
        block = cast(SpatialTransformerBlock, torch.compile(block, dynamic=False))
    with torch.no_grad():
        if features_last:
            output = block(x.moveaxis(1, -1), cond=cond).moveaxis(-1, 1)
        else:
            output = block(x, cond=cond)
    assert output.shape == x.shape
    assert not output.isnan().any(), 'NaN values in output'
