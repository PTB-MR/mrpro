"""Test Hourglass Transformer"""

from typing import cast

import pytest
import torch
from mr2.nn.nets import HourglassTransformer
from tests.conftest import minimal_torch_26


@minimal_torch_26
@torch.no_grad()
@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_hourglass_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the hourglass."""
    hourglass = HourglassTransformer(
        n_dim=2,
        n_channels_in=1,
        n_channels_out=1,
        depths=1,
        n_features=64,
        attention_neighborhood=(7, 7, None),
        cond_dim=32,
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    cond = torch.zeros(1, 32, device=device)
    hourglass = hourglass.to(device)
    x = x.to(device)
    cond = cond.to(device)
    if torch_compile:
        hourglass = cast(HourglassTransformer, torch.compile(hourglass, dynamic=False))
    y = hourglass(x, cond=cond)
    assert y.shape == (1, 1, 16, 16)


@minimal_torch_26
@pytest.mark.cuda
def test_hourglass_backward() -> None:
    hourglass = HourglassTransformer(
        n_dim=1,
        n_channels_in=1,
        n_channels_out=1,
        n_features=64,
        attention_neighborhood=(7, 7, None),
        cond_dim=32,
    ).cuda()

    x = torch.zeros(1, 1, 16, requires_grad=True).cuda()
    cond = torch.zeros(1, 32, requires_grad=True).cuda()
    y = hourglass(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in hourglass.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
