"""Variational Autoencoder with a Gaussian latent space."""

import torch
from torch.nn import Module


class VAE(Module):
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
        encoder : Module
            Encoder module. Should return double the number of channels of the latent space.
        decoder : Module
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
        x : torch.Tensor
            Input tensor

        Returns
        -------
            tuple of the reconstructed image and
            the KL divergence between the latent space and the standard normal distribution.
        """
        return self.forward(x)

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
        kl = -0.5 * torch.sum(1 + logvar - mean.square() - std.square())
        return reconstruction, kl
