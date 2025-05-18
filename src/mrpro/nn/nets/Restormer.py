"""Restormer implementation."""

from collections.abc import Sequence

import torch
from torch.nn import Module

from mrpro.nn.FiLM import FiLM
from mrpro.nn.NDModules import ConvND, ConvNd, InstanceNormNd
from mrpro.nn.nets.UNet import UNetBase
from mrpro.nn.PixelShuffle import PixelShuffle, PixelUnshuffle
from mrpro.nn.Sequential import Sequential
from mrpro.nn.TransposedAttention import TransposedAttention


class GDFN(Module):
    """Gated depthwise feed forward network.

    As used in the Restormer architecture.
    """

    def __init__(self, dim: int, channels: int, mlp_ratio: float):
        super().__init__()

        hidden_features = int(channels * mlp_ratio)
        self.project_in = ConvNd(dim)(channels, hidden_features * 2, kernel_size=1)
        self.depthwise_conv = ConvNd(dim)(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
        )
        self.project_out = ConvNd(dim)(hidden_features, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.depthwise_conv(x).chunk(2, dim=1)
        return self.project_out(torch.nn.functional.gelu(x1) * x2)


class RestormerBlock(Module):
    """Transformer block with transposed attention and gated depthwise feed forward network."""

    def __init__(self, dim: int, channels: int, num_heads: int, mlp_ratio: float, cond_dim: int = 0):
        super().__init__()
        self.norm1 = Sequential(InstanceNormNd(dim)(channels))
        self.attn = TransposedAttention(dim, channels, num_heads)
        self.norm2 = Sequential(InstanceNormNd(dim)(channels))
        self.ffn = GDFN(dim, channels, mlp_ratio)
        if cond_dim > 0:
            self.norm1.append(FiLM(channels=channels, cond_dim=cond_dim))
            self.norm2.append(FiLM(channels=channels, cond_dim=cond_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        super().__init__()

        self.first = ConvNd(dim)(channels_in, n_channels_per_head, kernel_size=3, stride=1, padding=1, bias=False)

        def blocks(n_heads: int, n_blocks: int):
            layers = Sequential(
                *(RestormerBlock(dim, n_channels_per_head, n_heads, mlp_ratio) for _ in range(n_blocks))
            )

            if cond_dim > 0 and n_blocks > 1:
                layers.insert(1, FiLM(channels=n_channels_per_head * n_heads, cond_dim=cond_dim))
            return layers

        for n_block, n_heads in zip(n_blocks, n_heads, strict=False):
            self.input_blocks.append(blocks(n_heads, n_block))
            self.output_blocks.append(blocks(n_heads, n_block))
            self.skip_blocks.append(Identity())

        self.output_blocks = self.output_blocks[::-1]
        for n_head_current, n_head_next in pairwise(n_heads):
            self.down_blocks.append(
                Sequential(
                    ConvND(dim)(
                        n_head_current * n_channels_per_head,
                        n_head_next * n_channels_per_head // 2**dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    PixelUnshuffle(2),
                )
            )
            self.up_blocks.append(
                Sequential(
                    ConvND(dim)(
                        n_head_next * n_channels_per_head,
                        n_head_current * n_channels_per_head * 2**dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    PixelShuffle(2),
                )
            )

        self.middle_block = blocks(n_heads, n_blocks)
        self.last = Sequential(
            *blocks(n_heads[0], n_refinement_blocks),
            ConvND(dim)(n_channels_per_head * n_heads[0], channels_out, kernel_size=3, stride=1, padding=1),
        )
