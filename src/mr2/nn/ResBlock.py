"""Residual convolution block with two convolutions."""

import torch
from torch.nn import Identity, Module, SiLU

from mr2.nn.CondMixin import CondMixin
from mr2.nn.FiLM import FiLM
from mr2.nn.GroupNorm import GroupNorm
from mr2.nn.ndmodules import convND
from mr2.nn.Sequential import Sequential


class ResBlock(CondMixin, Module):
    """Residual convolution block with two convolutions."""

    def __init__(self, n_dim: int, n_channels_in: int, n_channels_out: int, cond_dim: int) -> None:
        """Initialize the ResBlock.

        Parameters
        ----------
        n_dim
            The number of dimensions, i.e. 1, 2 or 3.
        n_channels_in
            The number of channels in the input tensor.
        n_channels_out
            The number of channels in the output tensor.
        cond_dim
            The number of features in the conditioning tensor used in a FiLM.
            If set to 0 no FiLM is used.

        """
        super().__init__()
        self.rezero = torch.nn.Parameter(torch.tensor(0.1))
        self.block = Sequential(
            GroupNorm(n_channels_in),
            SiLU(),
            convND(n_dim)(n_channels_in, n_channels_out, kernel_size=3, padding=1),
            GroupNorm(n_channels_out),
            SiLU(),
            convND(n_dim)(n_channels_out, n_channels_out, kernel_size=3, padding=1),
        )
        if cond_dim > 0:
            self.block.insert(1, FiLM(n_channels_in, cond_dim))
            self.block.insert(-2, FiLM(n_channels_out, cond_dim))

        if n_channels_out == n_channels_in:
            self.skip_connection: Module = Identity()
        else:
            self.skip_connection = convND(n_dim)(n_channels_in, n_channels_out, kernel_size=1)

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
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
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the ResBlock."""
        h = self.block(x, cond=cond)
        x = self.skip_connection(x) + self.rezero * h
        return x
