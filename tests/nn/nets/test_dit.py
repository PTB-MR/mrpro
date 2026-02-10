"""Tests for DiT network."""

from typing import cast

import pytest
import torch
from mrpro.nn.nets import DiT
from mrpro.nn.nets.DiT import DiTBlock
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_ditblock_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of DiTBlock."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 64, 32)).to(device).requires_grad_(True)
    cond = rng.float32_tensor((1, 16)).to(device).requires_grad_(True)
    block = DiTBlock(n_channels=32, n_heads=4, cond_dim=16, mlp_ratio=2.0, features_last=True).to(device)
    if torch_compile:
        block = cast(DiTBlock, torch.compile(block, dynamic=False))
    y = block(x, cond=cond)
    assert y.shape == x.shape
    assert not y.isnan().any(), 'NaN values in output'


def test_ditblock_backward() -> None:
    """Test the backward pass of DiTBlock."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 32, 8, 8)).requires_grad_(True)
    cond = rng.float32_tensor((1, 12)).requires_grad_(True)
    block = DiTBlock(n_channels=32, n_heads=4, cond_dim=12, mlp_ratio=2.0, features_last=False)
    y = block(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_dit_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of DiT."""
    dit = DiT(
        n_dim=2,
        n_channels_in=1,
        cond_dim=32,
        input_size=16,
        patch_size=2,
        n_channels_out=1,
        hidden_dim=32,
        depth=2,
        n_heads=4,
        mlp_ratio=2.0,
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    cond = torch.zeros(1, 32, device=device)
    dit = dit.to(device)
    if torch_compile:
        dit = cast(DiT, torch.compile(dit))
    y = dit(x, cond=cond)
    assert y.shape == (1, 1, 16, 16)


def test_dit_forward_3d() -> None:
    """Test DiT with 3D input."""
    dit = DiT(
        n_dim=3,
        n_channels_in=1,
        cond_dim=16,
        input_size=(8, 8, 8),
        patch_size=2,
        n_channels_out=1,
        hidden_dim=32,
        depth=1,
        n_heads=4,
        mlp_ratio=2.0,
    )
    x = torch.zeros(1, 1, 8, 8, 8)
    cond = torch.zeros(1, 16)
    y = dit(x, cond=cond)
    assert y.shape == (1, 1, 8, 8, 8)


def test_dit_backward() -> None:
    """Test the backward pass of DiT."""
    dit = DiT(
        n_dim=2,
        n_channels_in=1,
        cond_dim=24,
        input_size=16,
        patch_size=2,
        n_channels_out=1,
        hidden_dim=32,
        depth=2,
        n_heads=4,
        mlp_ratio=2.0,
    )

    x = torch.zeros(1, 1, 16, 16, requires_grad=True)
    cond = torch.zeros(1, 24, requires_grad=True)
    y = dit(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in dit.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
