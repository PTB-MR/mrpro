from collections.abc import Sequence

import torch
from torch.nn import Module, SiLU

from mrpro.nn.FiLM import FiLM
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.ndmodules import ConvND
from mrpro.nn.PermutedBlock import PermutedBlock
from mrpro.nn.Sequential import Sequential


class SeparableResBlock(Module):
    """Residual block with separable convolutions and ReZero."""

    def __init__(
        self,
        dim_groups: Sequence[Sequence[int]],
        channels_in: int,
        channels_out: int,
        cond_dim: int,
    ) -> None:
        """Initialize the SeparableResBlock.

        Applies convolutions as separable convolutions with SilU activation and group normalization.
        For example, if ``dim_groups = ((-1,-2), (-3))`` then one 2D convolution is applied to the last two dimensions,
        and one 1D convolution is applied to the last dimension.
        The order within the block is Norm->Activation->Conv.
        The whole sequence for all dimension groups is performed twice, with optional FiLM conditioning in between.
        So for two `dim_groups`, a total of 4 convolutions are applied.

        Parameters
        ----------
        dim_groups
            Sequence of dimension groups to use in the convolutions.
        channels_in
            Number of input channels.
        channels_out
            Number of output channels.
        cond_dim
            Number of channels in the conditioning tensor. If 0, no conditioning is applied.
        """
        super().__init__()
        self.rezero = torch.nn.Parameter(torch.tensor(0.1))

        def block(dims: Sequence[int], channels_in: int) -> Module:
            return Sequential(
                GroupNorm(channels_in),
                SiLU(),
                PermutedBlock(dims, ConvND(len(dims))(channels_in, channels_out, 3, padding=1)),
            )

        blocks = Sequential(*(block(d, channels_in if i == 0 else channels_out) for i, d in enumerate(dim_groups)))
        if cond_dim > 0:
            blocks.append(FiLM(channels_out, cond_dim))
        blocks.extend(block(d, channels_out) for d in dim_groups)
        self.block = blocks
        self.skip_connection = None
        if channels_in != channels_out:
            self.skip_connection = torch.nn.Linear(channels_in, channels_out)

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the SeparableResBlock.

        Parameters
        ----------
        x
            Input tensor.
        cond
            Conditioning tensor.

        Returns
        -------
            Output tensor with the same number and order of dimensions as the input.
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the SeparableResBlock."""
        h = self.block(x, cond=cond)
        if self.skip_connection is None:
            skip = x
        else:
            skip = torch.moveaxis(x, 1, -1)
            skip = self.skip_connection(skip)
            skip = torch.moveaxis(skip, -1, 1)
        return skip + self.rezero * h


from collections.abc import Sequence

import torch
from torch.nn import Module


class SeparableResBlock(Module):
    """Residual block with separable convolutions and ReZero."""

    def __init__(
        self,
        dim_groups: Sequence[Sequence[int]],
        channels_in: int,
        channels_out: int,
        cond_dim: int,
    ) -> None:
        """Initialize the SeparableResBlock.

        Applies convolutions as separable convolutions with SilU activation and group normalization.
        For example, if ``dim_groups = ((-1,-2), (-3))`` then one 2D convolution is applied to the last two dimensions,
        and one 1D convolution is applied to the last dimension.
        The order within the block is Norm->Activation->Conv.
        The whole sequence for all dimension groups is performed twice, with optional FiLM conditioning in between.
        So for two `dim_groups`, a total of 4 convolutions are applied.

        Parameters
        ----------
        dim_groups
            Sequence of dimension groups to use in the convolutions.
        channels_in
            Number of input channels.
        channels_out
            Number of output channels.
        cond_dim
            Number of channels in the conditioning tensor. If 0, no conditioning is applied.
        """
        super().__init__()
        self.rezero = torch.nn.Parameter(torch.tensor(0.1))

        def block(dims: Sequence[int], channels_in: int) -> Module:
            return Sequential(
                GroupNorm(channels_in),
                SiLU(),
                PermutedBlock(dims, ConvND(len(dims))(channels_in, channels_out, 3, padding=1)),
            )

        blocks = Sequential(*(block(d, channels_in if i == 0 else channels_out) for i, d in enumerate(dim_groups)))
        if cond_dim > 0:
            blocks.append(FiLM(channels_out, cond_dim))
        blocks.extend(block(d, channels_out) for d in dim_groups)
        self.block = blocks
        self.skip_connection = None
        if channels_in != channels_out:
            self.skip_connection = torch.nn.Linear(channels_in, channels_out)

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the SeparableResBlock.

        Parameters
        ----------
        x
            Input tensor.
        cond
            Conditioning tensor.

        Returns
        -------
            Output tensor with the same number and order of dimensions as the input.
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the SeparableResBlock."""
        h = self.block(x, cond=cond)
        if self.skip_connection is None:
            skip = x
        else:
            skip = torch.moveaxis(x, 1, -1)
            skip = self.skip_connection(skip)
            skip = torch.moveaxis(skip, -1, 1)
        return skip + self.rezero * h
