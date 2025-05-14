from collections.abc import Sequence
from itertools import pairwise

import torch
from sympy import Identity
from torch.nn import GELU, LeakyReLU, Module, Sequential

from mrpro.nn.NDModules import ConvND, ConvTransposeND, InstanceNormND
from mrpro.nn.nets.UNet import UNetBase
from mrpro.nn.ShiftedWindowAttention import ShiftedWindowAttention


class LeFF(Module):
    """Locally-enhanced Feed-Forward Network.

    Part of the Uformer architecture.
    """

    def __init__(
        self,
        dim: int,
        channels_in: int = 32,
        channels_out: int = 32,
        expand_ratio: float = 4,
    ) -> None:
        """Initialize the LeFF module.

        Parameters
        ----------
        dim : int
            2 or 3, for 2D or 3D input
        channels_in : int
            Input feature dimension
        channels_out : int
            Output feature dimension
        expand_ratio : float
            Expansion ratio of the hidden dimension
        """
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.block = Sequential(
            ConvND(dim)(channels_in, hidden_dim, 1),
            GELU(),
            ConvND(dim)(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, stride=1, padding=1),
            GELU(),
            ConvND(dim)(hidden_dim, channels_out, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LeWinTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        n_channels_per_head: int,
        n_heads: int,
        window_size: int = 8,
        shifted: bool = False,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        channels = n_channels_per_head * n_heads
        self.norm1 = InstanceNormND(dim)(channels)
        self.attn = ShiftedWindowAttention(
            dim=dim,
            n_channels_per_head=n_channels_per_head,
            n_heads=n_heads,
            window_size=window_size,
            shifted=shifted,
        )

        self.norm2 = InstanceNormND(dim)(channels)
        self.ff = LeFF(dim=dim, channels_in=channels, channels_out=channels, expand_ratio=mlp_ratio)
        self.modulator = torch.nn.Parameter(torch.empty(channels, *((window_size,) * dim)))
        torch.nn.init.trunc_normal_(self.modulator)

    def forward(self, x):
        modulator = self.modulator.tile([t // s for t, s in zip(x.shape[1:], self.modulator.shape, strict=False)])
        x_mod = self.norm1(x) + modulator
        x_attn = self.attn(x_mod)
        x_ff = self.ff(self.norm2(x_attn))
        return x + x_ff


class Uformer(UNetBase):
    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        n_features_per_head: int = 32,
        n_heads: Sequence[int] = (1, 2, 4, 8),
        n_blocks: int = 2,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        def blocks(n_heads: int):
            return [
                LeWinTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_features_per_head=n_features_per_head,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    shifted=bool(i % 2),
                )
                for i in range(n_blocks)
            ]

        for n_head in n_heads:
            self.input_blocks.extend(blocks(n_heads=n_head))
            self.output_blocks.extend(blocks(n_heads=n_head))
            self.skip_blocks.append(Identity())
        self.middle_block = torch.nn.Sequential(*blocks(n_heads=n_heads[-1]))

        for n_head_current, n_head_next in pairwise(n_heads):
            self.down_blocks.append(
                ConvND(dim)(
                    n_features_per_head * n_head_current,
                    n_features_per_head * n_head_next,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.up_blocks.append(
                ConvTransposeND(dim)(
                    n_features_per_head * n_head_next, n_features_per_head * n_head_current, kernel_size=2, stride=2
                )
            )
        self.first = torch.nn.Sequential(
            ConvND(dim)(channels_in, n_features_per_head * n_heads[0], kernel_size=3, stride=1, padding='same'),
            LeakyReLU(),
        )
        self.last = ConvND(dim)(
            n_features_per_head * n_heads[-1], channels_out, kernel_size=3, stride=1, padding='same'
        )
