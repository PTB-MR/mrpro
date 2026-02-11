"""Restormer implementation."""

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch.nn import Module

from mr2.nn.attention.TransposedAttention import TransposedAttention
from mr2.nn.CondMixin import CondMixin
from mr2.nn.FiLM import FiLM
from mr2.nn.join import Concat
from mr2.nn.ndmodules import convND, instanceNormND
from mr2.nn.nets.UNet import UNetBase, UNetDecoder, UNetEncoder
from mr2.nn.PixelShuffle import PixelShuffleUpsample, PixelUnshuffleDownsample
from mr2.nn.Sequential import Sequential


class GDFN(Module):
    """Gated depthwise feed forward network.

    Feed-forward block used in Restormer [ZAM22]_. It first expands channels,
    applies a depthwise convolution, then uses a gated interaction between two
    channel splits before projecting back to the input width.

    References
    ----------
    .. [ZAM22] Zamir, Syed Waqas, et al. "Restormer: Efficient transformer for
       high-resolution image restoration." CVPR 2022.
    """

    def __init__(self, n_dim: int, n_channels: int, mlp_ratio: float):
        """Initialize GDFN.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels
            The number of channels in the input tensor.
        mlp_ratio
            Ratio for hidden dimension expansion
        """
        super().__init__()

        hidden_features = int(n_channels * mlp_ratio)
        self.project_in = convND(n_dim)(n_channels, hidden_features * 2, kernel_size=1)
        self.depthwise_conv = convND(n_dim)(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
        )
        self.project_out = convND(n_dim)(hidden_features, n_channels, kernel_size=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated depthwise feed forward network.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Output tensor
        """
        x = self.project_in(x)
        x1, x2 = self.depthwise_conv(x).chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.project_out(x)
        return x


class RestormerBlock(CondMixin, Module):
    """Transformer block with transposed attention and gated depthwise feed forward network."""

    def __init__(self, n_dim: int, n_channels: int, n_heads: int, mlp_ratio: float, cond_dim: int = 0):
        """Initialize RestormerBlock.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels
            The number of channels in the input tensor.
        n_heads
            Number of attention heads
        mlp_ratio
            Ratio for hidden dimension expansion
        cond_dim
            Dimension of conditioning input. If 0, no conditioning is applied.
        """
        super().__init__()
        self.norm1 = Sequential(instanceNormND(n_dim)(n_channels))
        self.attn = TransposedAttention(n_dim, n_channels, n_channels, n_heads)
        self.norm2 = Sequential(instanceNormND(n_dim)(n_channels))
        self.ffn = GDFN(n_dim, n_channels, mlp_ratio)
        if cond_dim > 0:
            self.norm2.append(FiLM(channels=n_channels, cond_dim=cond_dim))

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply Restormer block.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning tensor. If None, no conditioning is applied.

        Returns
        -------
            Output tensor
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for RestormerBlock."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x, cond=cond))
        return x


class Restormer(UNetBase):
    """Restormer architecture.

    Implements the Restormer [ZAM22]_ network, which is a U-shaped transformer
    with channel wise attention and depthwise convolutions in the feed forward network.

    References
    ----------
    .. [ZAM22] Zamir, Syed Waqas, et al. "Restormer: Efficient transformer for high-resolution image restoration."
       CVPR 2022, https://arxiv.org/pdf/2111.09881.pdf
    """

    def __init__(
        self,
        n_dim: int,
        n_channels_in: int,
        n_channels_out: int,
        n_blocks: Sequence[int] = (4, 6, 6, 8),
        n_refinement_blocks: int = 4,
        n_heads: Sequence[int] = (1, 2, 4, 8),
        n_channels_per_head: int = 48,
        mlp_ratio: float = 2.66,
        cond_dim: int = 0,
    ):
        """Initialize Restormer.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels_in
            The number of input channels.
        n_channels_out
            The number of output channels.
        n_blocks
            Number of blocks in each stage
        n_refinement_blocks
            Number of refinement blocks
        n_heads
            Number of attention heads in each stage
        n_channels_per_head
            Number of channels per attention head
        mlp_ratio
            Ratio for hidden dimension expansion
        cond_dim
            Dimension of conditioning input. If 0, no conditioning is applied.
        """
        if len(n_blocks) != len(n_heads):
            raise ValueError('n_blocks and n_heads must have the same length.')

        def blocks(n_heads: int, n_blocks: int):
            layers = Sequential(
                *(RestormerBlock(n_dim, n_channels_per_head * n_heads, n_heads, mlp_ratio) for _ in range(n_blocks))
            )

            if cond_dim > 0 and n_blocks > 1:
                layers.insert(1, FiLM(channels=n_channels_per_head * n_heads, cond_dim=cond_dim))
            return layers

        first_block = convND(n_dim)(n_channels_in, n_channels_per_head, kernel_size=3, stride=1, padding=1, bias=False)
        encoder_blocks = [blocks(head, block) for head, block in zip(n_heads[:-1], n_blocks[:-1], strict=True)]
        down_blocks = [
            PixelUnshuffleDownsample(n_dim, n_channels_per_head * head_current, n_channels_per_head * head_next)
            for head_current, head_next in pairwise(n_heads)
        ]
        middle_block = blocks(n_heads[-1], n_blocks[-1])
        encoder = UNetEncoder(
            first_block=first_block,
            blocks=encoder_blocks,
            down_blocks=down_blocks,
            middle_block=middle_block,
        )

        up_blocks = [
            PixelShuffleUpsample(n_dim, n_channels_per_head * head_next, n_channels_per_head * head_current)
            for head_current, head_next in pairwise(n_heads)
        ][::-1]
        concat_blocks = [
            Sequential(
                Concat(),
                convND(n_dim)(2 * n_channels_per_head * head, n_channels_per_head * head, kernel_size=1),
            )
            for head in n_heads[-2::-1]
        ]
        decoder_blocks = [blocks(head, block) for head, block in zip(n_heads[:-1], n_blocks[:-1], strict=True)][::-1]
        last_block = Sequential(
            *(RestormerBlock(n_dim, n_channels_per_head, n_heads[0], mlp_ratio) for _ in range(n_refinement_blocks)),
            convND(n_dim)(n_channels_per_head, n_channels_out, kernel_size=3, stride=1, padding=1),
        )
        decoder = UNetDecoder(
            blocks=decoder_blocks,
            up_blocks=up_blocks,
            concat_blocks=concat_blocks,
            last_block=last_block,
        )

        super().__init__(encoder=encoder, decoder=decoder)
