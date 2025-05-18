"""Residual convolution block with two convolutions."""

import torch
from torch.nn import Identity, Module, SiLU

from mrpro.nn.CondMixin import CondMixin
from mrpro.nn.FiLM import FiLM
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.NDModules import ConvND
from mrpro.nn.Sequential import Sequential


class ResBlock(CondMixin, Module):
    """Residual convolution block with two convolutions."""

    def __init__(self, dim: int, channels_in: int, channels_out: int, cond_dim: int) -> None:
        """Initialize the ResBlock.

        Parameters
        ----------
        dim
            The dimension, i.e. 1, 2 or 3.
        channels_in
            The number of channels in the input tensor.
        channels_out
            The number of channels in the output tensor.
        cond_dim
            The number of features in the conditioning tensor used in a FiLM.
            If set to 0 no FiLM is used.

        """
        super().__init__()
        self.rezero = torch.nn.Parameter(torch.tensor(1e-6))
        self.block = Sequential(
            GroupNorm(channels_in),
            SiLU(),
            ConvND(dim)(channels_in, channels_out, kernel_size=3, padding=1),
            GroupNorm(channels_out),
            SiLU(),
            ConvND(dim)(channels_out, channels_out, kernel_size=3, padding=1),
        )
        if cond_dim > 0:
            self.block.insert(-3, FiLM(channels_out, cond_dim))

        if channels_out == channels_in:
            self.skip_connection: Module = Identity()
        else:
            self.skip_connection = ConvND(dim)(channels_in, channels_out, kernel_size=1)

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the ResBlock.

        Parameters
        ----------
        x
            The input tensor.
        cond
            A conditioning tensor to be used for FiLM.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the ResBlock."""
        h = self.block(x, cond)
        x = self.skip_connection(x) + h
        return x
