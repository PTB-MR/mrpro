"""Restormer implementation."""

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch.nn import Module

from mrpro.nn.FiLM import FiLM
from mrpro.nn.join import Concat
from mrpro.nn.ndmodules import ConvND, InstanceNormND
from mrpro.nn.nets.UNet import UNetBase, UNetDecoder, UNetEncoder
from mrpro.nn.PixelShuffle import PixelShuffleUpsample, PixelUnshuffleDownsample
from mrpro.nn.Sequential import Sequential
from mrpro.nn.TransposedAttention import TransposedAttention


class GDFN(Module):
    """Gated depthwise feed forward network.

    As used in the Restormer architecture.
    """

    def __init__(self, dim: int, channels: int, mlp_ratio: float):
        """Initialize GDFN.

        Parameters
        ----------
        dim : int
            Dimension of the input space
        channels : int
            Number of input/output channels
        mlp_ratio : float
            Ratio for hidden dimension expansion
        """
        super().__init__()

        hidden_features = int(channels * mlp_ratio)
        self.project_in = ConvND(dim)(channels, hidden_features * 2, kernel_size=1)
        self.depthwise_conv = ConvND(dim)(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
        )
        self.project_out = ConvND(dim)(hidden_features, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated depthwise feed forward network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = self.project_in(x)
        x1, x2 = self.depthwise_conv(x).chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.project_out(x)
        return x


class RestormerBlock(Module):
    """Transformer block with transposed attention and gated depthwise feed forward network."""

    def __init__(self, dim: int, channels: int, n_heads: int, mlp_ratio: float, cond_dim: int = 0):
        """Initialize RestormerBlock.

        Parameters
        ----------
        dim : int
            Dimension of the input space
        channels : int
            Number of input/output channels
        n_heads : int
            Number of attention heads
        mlp_ratio : float
            Ratio for hidden dimension expansion
        cond_dim : int, optional
            Dimension of conditioning input
        """
        super().__init__()
        self.norm1 = Sequential(InstanceNormND(dim)(channels))
        self.attn = TransposedAttention(dim, channels, channels, n_heads)
        self.norm2 = Sequential(InstanceNormND(dim)(channels))
        self.ffn = GDFN(dim, channels, mlp_ratio)
        if cond_dim > 0:
            self.norm2.append(FiLM(channels=channels, cond_dim=cond_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Restormer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
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
        dim: int,
        channels_in: int,
        channels_out: int,
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
        dim : int
            Dimension of the input space
        channels_in : int
            Number of input channels
        channels_out : int
            Number of output channels
        n_blocks : Sequence[int], optional
            Number of blocks in each stage
        n_refinement_blocks : int, optional
            Number of refinement blocks
        n_heads : Sequence[int], optional
            Number of attention heads in each stage
        n_channels_per_head : int, optional
            Number of channels per attention head
        mlp_ratio : float, optional
            Ratio for hidden dimension expansion
        cond_dim : int, optional
            Dimension of conditioning input
        """

        def blocks(n_heads: int, n_blocks: int):
            layers = Sequential(
                *(RestormerBlock(dim, n_channels_per_head * n_heads, n_heads, mlp_ratio) for _ in range(n_blocks))
            )

            if cond_dim > 0 and n_blocks > 1:
                layers.insert(1, FiLM(channels=n_channels_per_head * n_heads, cond_dim=cond_dim))
            return layers

        first_block = ConvND(dim)(channels_in, n_channels_per_head, kernel_size=3, stride=1, padding=1, bias=False)
        encoder_blocks = [blocks(head, block) for head, block in zip(n_heads[:-1], n_blocks[:-1], strict=True)]
        down_blocks = [
            PixelUnshuffleDownsample(dim, n_channels_per_head * head_current, n_channels_per_head * head_next)
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
            PixelShuffleUpsample(dim, n_channels_per_head * head_next, n_channels_per_head * head_current)
            for head_current, head_next in pairwise(n_heads)
        ][::-1]
        concat_blocks = [Concat() for _ in range(len(encoder_blocks))]
        decoder_blocks = [blocks(head, block) for head, block in zip(n_heads[:-1], n_blocks[:-1], strict=True)][::-1]
        last_block = Sequential(
            *(RestormerBlock(dim, n_channels_per_head, n_heads[0], mlp_ratio) for _ in range(n_refinement_blocks)),
            ConvND(dim)(n_channels_per_head, channels_out, kernel_size=3, stride=1, padding=1),
        )
        decoder = UNetDecoder(
            blocks=decoder_blocks,
            up_blocks=up_blocks,
            concat_blocks=concat_blocks,
            last_block=last_block,
        )

        super().__init__(encoder=encoder, decoder=decoder)
