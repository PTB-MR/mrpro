from typing import cast

import pytest
import torch
from mrpro.nn.nets import UNet


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
        dim=2,
        channels_in=1,
        channels_out=1,
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
