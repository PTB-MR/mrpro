"""Variational Autoencoder with a Gaussian latent space."""

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch.nn import Module, SiLU

from mr2.nn.GroupNorm import GroupNorm
from mr2.nn.ndmodules import convND
from mr2.nn.ResBlock import ResBlock
from mr2.nn.Sequential import Sequential
from mr2.nn.Upsample import Upsample


class VAEBase(Module):
    """Basic Variational Autoencoder.

    Consists of an encoder to transform the input into a latent space and a decoder to transform the latent space back
    into the original space. The encoder should return twice the number of channels as the decoder needs to reconstruct
    the input: half of the channels are the mean and the other half the log variance of the latent space.
    The reparameterization trick is used to sample from the latent space.
    The forward pass returns the reconstructed image and the KL divergence between the latent space and the standard
    normal distribution.
    """

    def __init__(self, encoder: Module, decoder: Module):
        """Initialize the VAE.

        Parameters
        ----------
        encoder
            Encoder module. Should return double the number of channels of the latent space.
        decoder
            Decoder module
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE.

        Calculates the reconstruction as well as the KL divergence between the latent space and the
        standard normal distribution.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            tuple of the reconstructed image and
            the KL divergence between the latent space and the standard normal distribution.
        """
        return super().__call__(x)

    def mode(self, x: torch.Tensor) -> torch.Tensor:
        """Mode of the VAE."""
        z = self.encoder(x)
        mean, _ = z.chunk(2, dim=1)
        return self.decoder(mean)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE."""
        z = self.encoder(x)
        mean, logvar = z.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + torch.randn_like(std) * std
        reconstruction = self.decoder(sample)
        kl = (-0.5 / len(z)) * torch.sum(1 + logvar - mean.square() - std.square())
        return reconstruction, kl


class VAE(VAEBase):
    """Variational autoencoder with convolutional encoder and decoder."""

    def __init__(
        self,
        n_dim: int = 2,
        n_channels_in: int = 2,
        latent_channels: int = 8,
        n_features: Sequence[int] = (32, 64, 128),
        n_res_blocks: int = 2,
    ) -> None:
        """Initialize the VAE.

        Parameters
        ----------
        n_dim
            The number of dimensions, i.e. 1, 2 or 3.
        n_channels_in
            The number of channels in the input tensor.
        latent_channels
            The number of channels in the latent space.
        n_features
            The number of features at each resolution level.
        n_res_blocks
            Number of residual blocks per resolution level.
        """
        encoder = Sequential(convND(n_dim)(n_channels_in, n_features[0], kernel_size=3, padding=1))

        for n_feat, n_feat_next in pairwise(n_features):
            for _ in range(n_res_blocks):
                encoder.append(ResBlock(n_dim, n_feat, n_feat, cond_dim=0))
            encoder.append(convND(n_dim)(n_feat, n_feat_next, kernel_size=3, stride=2, padding=1))

        for _ in range(n_res_blocks):
            encoder.append(ResBlock(n_dim, n_features[-1], n_features[-1], cond_dim=0))

        encoder.extend(
            [
                GroupNorm(n_features[-1]),
                SiLU(),
                convND(n_dim)(n_features[-1], 2 * latent_channels, kernel_size=3, padding=1),
            ]
        )

        decoder = Sequential(convND(n_dim)(latent_channels, n_features[-1], kernel_size=3, padding=1))
        for _ in range(n_res_blocks):
            decoder.append(ResBlock(n_dim, n_features[-1], n_features[-1], cond_dim=0))

        for n_feat, n_feat_next in pairwise(reversed(n_features)):
            decoder.append(
                Sequential(
                    Upsample(dim=range(-n_dim, 0), scale_factor=2, mode='linear'),
                    convND(n_dim)(n_feat, n_feat_next, kernel_size=3, padding=1),
                )
            )
            for _ in range(n_res_blocks):
                decoder.append(ResBlock(n_dim, n_feat_next, n_feat_next, cond_dim=0))

        decoder.extend(
            [
                GroupNorm(n_features[0]),
                SiLU(),
                convND(n_dim)(n_features[0], n_channels_in, kernel_size=3, padding=1),
            ]
        )

        super().__init__(encoder=encoder, decoder=decoder)
