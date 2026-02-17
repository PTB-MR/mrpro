"""Tests for VAE network."""

from typing import cast

import pytest
import torch
from mr2.nn.nets import VAE


@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_vae_forward(torch_compile: bool, device: str) -> None:
    """Test the forward pass of the VAE."""
    vae = VAE(
        n_dim=2,
        n_channels_in=1,
        latent_channels=4,
        n_features=(2, 4),
        n_res_blocks=2,
    )

    x = torch.zeros(1, 1, 8, 8, device=device)
    vae = vae.to(device)
    x = x.to(device)
    if torch_compile:
        vae = cast(VAE, torch.compile(vae))
    y, kl = vae(x)
    assert y.shape == (1, 1, 8, 8)
    assert kl.shape == ()
    latent = vae.encoder(x)
    assert latent.shape == (1, 2 * 4, 4, 4)  # 2 because of mean and logvar


def test_vae_backward_kl() -> None:
    """Test the backward pass of the VAE wrt kl."""
    vae = VAE(
        n_dim=1,
        n_channels_in=1,
        latent_channels=4,
        n_features=(6, 8, 10),
        n_res_blocks=2,
    )

    x = torch.zeros(1, 1, 8, requires_grad=True)

    _, kl = vae(x)
    kl.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    for name, parameter in vae.encoder.named_parameters():  # only the encoder parameters can influence kl
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'


def test_vae_backward_y() -> None:
    """Test the backward pass of the VAE wrt y."""
    vae = VAE(
        n_dim=1,
        n_channels_in=1,
        latent_channels=4,
        n_features=(6, 8, 10),
        n_res_blocks=2,
    )

    x = torch.zeros(1, 1, 8, requires_grad=True)

    y, _ = vae(x)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    for name, parameter in vae.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
