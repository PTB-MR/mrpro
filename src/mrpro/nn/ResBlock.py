"""Residual convolution block with two convolutions."""

import torch
from torch.nn import Identity, Module, Sequential, SiLU

from mrpro.nn.NDModules import ConvND
from mrpro.nn.EmbMixin import EmbMixin
from mrpro.nn.GroupNorm32 import GroupNorm32
from mrpro.nn.FiLM import FiLM


class ResBlock(Module, EmbMixin):
    """Residual convolution block with two convolutions."""

    def __init__(self, dim: int, channels_in: int, channels_out: int, channels_emb: int) -> None:
        """Initialize the ResBlock.

        Parameters
        ----------
        dim
            The dimension, i.e. 1, 2 or 3.
        channels_in
            The number of channels in the input tensor.
        channels_out
            The number of channels in the output tensor.
        channels_emb
            The number of channels in the embedding tensor used in a FiLM embedding.
            If set to 0 no FiLM embedding is used.

        """
        super().__init__()
        self.rezero = torch.nn.Parameter(torch.tensor(1e-6))
        self.block = Sequential(
            GroupNorm32(channels_in),
            SiLU(),
            ConvND(dim)(channels_in, channels_out, kernel_size=3),
            GroupNorm32(channels_out),
            SiLU(),
            ConvND(dim)(channels_out, channels_out, kernel_size=3),
        )
        if channels_emb > 0:
            self.block.insert(-3, FiLM(channels_out, channels_emb))

        if channels_out == channels_in:
            self.skip_connection: Module = Identity()
        else:
            self.skip_connection = ConvND(dim)(channels_in, channels_out, kernel_size=1)

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        """Apply the ResBlock.

        Parameters
        ----------
        x
            The input tensor.
        emb
            An embedding tensor to be used for FiLM.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, emb)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        """Apply the ResBlock."""
        h = self.block(x, emb)
        x = self.skip_connection(x) + h
        return x
