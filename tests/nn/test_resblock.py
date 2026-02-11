"""Tests for ResBlock module."""

from collections.abc import Sequence
from typing import cast

import pytest
import torch
from mr2.nn import ResBlock
from mr2.utils import RandomGenerator


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'eager'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('dim', 'channels_in', 'channels_out', 'cond_dim', 'input_shape', 'cond_shape'),
    [
        (2, 32, 32, 16, (1, 32, 32, 32), (1, 16)),
        (3, 64, 32, 0, (2, 64, 16, 16, 16), None),
    ],
)
def test_resblock(
    dim: int,
    channels_in: int,
    channels_out: int,
    cond_dim: int,
    input_shape: Sequence[int],
    cond_shape: Sequence[int] | None,
    device: str,
    torch_compile: bool,
) -> None:
    """Test ResBlock output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    cond = rng.float32_tensor(cond_shape).to(device).requires_grad_(True) if cond_shape else None
    block = ResBlock(n_dim=dim, n_channels_in=channels_in, n_channels_out=channels_out, cond_dim=cond_dim).to(device)
    if torch_compile:
        block = cast(ResBlock, torch.compile(block, dynamic=False))
    output = block(x, cond=cond)
    assert output.shape == (input_shape[0], channels_out, *input_shape[2:]), (
        f'Output shape {output.shape} != expected {(input_shape[0], channels_out, *input_shape[2:])}'
    )
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert block.block[-1].weight.grad is not None, 'No gradient computed for first Conv'
    if cond is not None:
        assert cond.grad is not None, 'No gradient computed for conditioning'
        assert not cond.isnan().any(), 'NaN values in conditioning'
        assert not cond.grad.isnan().any(), 'NaN values in conditioning gradients'
