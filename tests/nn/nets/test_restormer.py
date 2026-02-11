"""Tests for Restormer network."""

from typing import cast

import pytest
import torch
from mr2.nn.nets import Restormer


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_restormer_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the restormer."""
    restormer = Restormer(
        n_dim=2,
        n_channels_in=1,
        n_channels_out=1,
        n_heads=(1, 2, 4),
        n_blocks=(2, 1, 1),
        cond_dim=32,
        n_channels_per_head=2,
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    cond = torch.zeros(1, 32, device=device)
    restormer = restormer.to(device)
    x = x.to(device)
    cond = cond.to(device)
    if torch_compile:
        restormer = cast(Restormer, torch.compile(restormer))
    y = restormer(x, cond=cond)
    assert y.shape == (1, 1, 16, 16)


def test_restormer_backward() -> None:
    restormer = Restormer(
        n_dim=1,
        n_channels_in=1,
        n_channels_out=1,
        n_heads=(1, 2),
        n_blocks=(2, 2),
        cond_dim=32,
        n_channels_per_head=4,
    )

    x = torch.zeros(1, 1, 16, requires_grad=True)
    cond = torch.zeros(1, 32, requires_grad=True)
    y = restormer(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in restormer.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
