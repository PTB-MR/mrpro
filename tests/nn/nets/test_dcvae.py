from typing import cast

import pytest
import torch
from mrpro.nn.nets import DCVAE


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_dcvae_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the DCVAE."""
    dcvae = DCVAE(
        n_dim=2,
        n_channels=1,
        latent_dim=4,
        block_types=('CNN', 'LinearViT', 'ViT'),
        widths=(32, 64, 32),
        depths=(2, 2, 3),
    )

    x = torch.zeros(1, 1, 16, 16, device=device)
    dcvae = dcvae.to(device)
    x = x.to(device)
    if torch_compile:
        dcvae = cast(DCVAE, torch.compile(dcvae))
    y, kl = dcvae(x)
    assert y.shape == (1, 1, 16, 16)
    assert kl.shape == ()
    latent = dcvae.encoder(x)
    assert latent.shape == (1, 2 * 4, 2, 2)  # 2 because of mean and logvar


def test_dcvae_backward():
    """Test the backward pass of the DCVAE."""
    dcvae = DCVAE(
        n_dim=1,
        n_channels=1,
        latent_dim=4,
        block_types=('CNN', 'LinearViT', 'ViT'),
        widths=(8, 12, 16),
        depths=(2, 2, 3),
    )

    x = torch.zeros(1, 1, 16, requires_grad=True)

    y, kl = dcvae(x)
    y.sum().backward()
    kl.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    for name, parameter in dcvae.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
