"""Tests for Mlp module."""

from typing import cast

import pytest
import torch
from mr2.nn.nets import MLP
from mr2.utils import RandomGenerator


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_mlp_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the Mlp."""
    mlp = MLP(
        n_channels_in=8,
        n_channels_out=4,
        norm='layer',
        activation='gelu',
        n_features=(16,),
        cond_dim=12,
        features_last=False,
    ).to(device)
    x = torch.zeros(1, 8, 9, 7, device=device)
    cond = torch.zeros(1, 12, device=device)
    if torch_compile:
        mlp = cast(MLP, torch.compile(mlp))
    y = mlp(x, cond=cond)
    assert y.shape == (1, 4, 9, 7)


def test_mlp_backward() -> None:
    """Test the backward pass of the Mlp."""
    mlp = MLP(
        n_channels_in=6,
        n_channels_out=3,
        norm='none',
        activation='silu',
        n_features=(12, 12),
        cond_dim=10,
        features_last=True,
    )
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 20, 6)).requires_grad_(True)
    cond = rng.float32_tensor((1, 10)).requires_grad_(True)
    y = mlp(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in mlp.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'


def test_mlp_features_last() -> None:
    """Test Mlp with features_last=True vs features_last=False."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).requires_grad_(True)

    mlp_last = MLP(
        n_channels_in=3,
        n_channels_out=4,
        norm='layer',
        activation='relu',
        n_features=(6,),
        cond_dim=0,
        features_last=True,
    )
    mlp = MLP(
        n_channels_in=3,
        n_channels_out=4,
        norm='layer',
        activation='relu',
        n_features=(6,),
        cond_dim=0,
        features_last=False,
    )
    mlp.load_state_dict(mlp_last.state_dict())
    y_last = mlp_last(x.moveaxis(1, -1))
    y = mlp(x)
    torch.testing.assert_close(y, y_last.moveaxis(-1, 1))
