"""Tests for UNet and AttentionGatedUNet networks."""

from typing import cast

import pytest
import torch
from mr2.nn.nets import AttentionGatedUNet, UNet


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_unet_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the UNet."""
    unet = UNet(
        n_dim=2,
        n_channels_in=1,
        n_channels_out=1,
        attention_depths=(-1,),
        n_features=(4, 6, 8),
        n_heads=2,
        cond_dim=32,
        encoder_blocks_per_scale=1,
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    cond = torch.zeros(1, 32, device=device)
    unet = unet.to(device)
    x = x.to(device)
    cond = cond.to(device)
    if torch_compile:
        unet = cast(UNet, torch.compile(unet))
    y = unet(x, cond=cond)
    assert y.shape == (1, 1, 16, 16)


def test_unet_backward() -> None:
    unet = UNet(
        n_dim=1,
        n_channels_in=1,
        n_channels_out=1,
        attention_depths=(-1,),
        n_features=(4, 6, 8),
        n_heads=2,
        cond_dim=32,
        encoder_blocks_per_scale=1,
    )

    x = torch.zeros(1, 1, 16, requires_grad=True)
    cond = torch.zeros(1, 32, requires_grad=True)
    y = unet(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in unet.named_parameters():
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
def test_gated_unet_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the AttentionGatedUNet."""
    unet = AttentionGatedUNet(
        n_dim=2,
        n_channels_in=1,
        n_channels_out=1,
        n_features=(4, 6, 8),
        cond_dim=32,
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    cond = torch.zeros(1, 32, device=device)
    unet = unet.to(device)
    x = x.to(device)
    cond = cond.to(device)
    if torch_compile:
        unet = cast(AttentionGatedUNet, torch.compile(unet))
    y = unet(x, cond=cond)
    assert y.shape == (1, 1, 16, 16)


def test_gated_unet_backward() -> None:
    """Test the backward pass of the AttentionGatedUNet."""
    unet = AttentionGatedUNet(
        n_dim=1,
        n_channels_in=1,
        n_channels_out=1,
        n_features=(4, 6, 8),
        cond_dim=32,
    )

    x = torch.zeros(1, 1, 16, requires_grad=True)
    cond = torch.zeros(1, 32, requires_grad=True)
    y = unet(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in unet.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
