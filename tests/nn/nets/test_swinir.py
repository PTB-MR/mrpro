"""Tests for SwinIR network."""

from typing import cast

import pytest
import torch
from mr2.nn.nets import SwinIR


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_swinir_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the UNet."""
    swinir = SwinIR(
        n_dim=2,
        n_channels_in=1,
        n_channels_out=1,
        n_heads=2,
        n_channels_per_head=4,
        n_blocks=2,
        window_size=4,
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    swinir = swinir.to(device)
    if torch_compile:
        swinir = cast(SwinIR, torch.compile(swinir))
    y = swinir(x)
    assert y.shape == (1, 1, 16, 16)


def test_swinir_backward() -> None:
    swinir = SwinIR(
        n_dim=1,
        n_channels_in=1,
        n_channels_out=1,
        n_heads=2,
        n_channels_per_head=4,
        n_blocks=2,
        window_size=4,
    )

    x = torch.zeros(1, 1, 16, requires_grad=True)
    y = swinir(x)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    for name, parameter in swinir.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
