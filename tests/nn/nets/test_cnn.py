from typing import cast

import pytest
import torch
from mrpro.nn.nets import BasicCNN


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_cnn_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the cnn."""
    cnn = BasicCNN(
        n_dim=2,
        n_channels_in=1,
        n_channels_out=1,
        norm='layer',
        n_features=(8, 8),
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    cond = torch.zeros(1, 32, device=device)
    cnn = cnn.to(device)
    x = x.to(device)
    cond = cond.to(device)
    if torch_compile:
        cnn = cast(BasicCNN, torch.compile(cnn))
    y = cnn(x, cond=cond)
    assert y.shape == (1, 1, 16, 16)
    assert y.mean().abs() < 0.1


def test_cnn_backward():
    cnn = BasicCNN(
        n_dim=1,
        n_channels_in=1,
        n_channels_out=1,
        norm='instance',
        activation='silu',
        n_features=(8, 8),
    )

    x = torch.zeros(1, 1, 16, requires_grad=True)
    cond = torch.zeros(1, 32, requires_grad=True)
    y = cnn(x, cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in cnn.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
