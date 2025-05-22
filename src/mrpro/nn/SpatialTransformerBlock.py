"""Spatial transformer block."""

import torch
from torch.nn import Dropout, Linear, Module

from mrpro.nn.GEGLU import GEGLU
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.LayerNorm import LayerNorm
from mrpro.nn.MultiHeadAttention import MultiHeadAttention
from mrpro.nn.ndmodules import ConvND
from mrpro.nn.Sequential import Sequential


def zero_init(m: Module) -> Module:
    """Initialize module weights and bias to zero."""
    if hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
        torch.nn.init.zeros_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
        torch.nn.init.zeros_(m.bias)
    return m


class BasicTransformerBlock(Module):
    def __init__(self, channels: int, n_heads: int, p_dropout: float = 0.0, cond_dim: int = 0, mlp_ratio: float = 4):
        super().__init__()
        self.selfattention = Sequential(
            LayerNorm(channels),
            MultiHeadAttention(channels_in=channels, channels_out=channels, n_heads=n_heads, p_dropout=p_dropout),
        )
        hidden_dim = int(channels * mlp_ratio)
        self.ff = Sequential(
            LayerNorm(channels), GEGLU(channels, hidden_dim), Dropout(p_dropout), Linear(hidden_dim, channels)
        )
        self.crossattention = (
            Sequential(
                LayerNorm(channels),
                MultiHeadAttention(
                    channels_in=channels,
                    channels_out=channels,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    channels_cross=cond_dim,
                ),
            )
            if cond_dim > 0
            else None
        )
        self.norm2 = LayerNorm(channels)
        self.cond_dim = cond_dim

    def forward(self, x, cond: torch.Tensor | None = None):
        x = self.selfattention(x) + x
        if cond is not None and self.crossattention is not None:
            cond = cond.unflatten(-1, (-1, self.cond_dim))
            x = self.crossattention(x, cond=cond) + x
        x = self.ff(x) + x
        return x


class SpatialTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        channels: int,
        n_heads: int,
        channels_per_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = channels
        hidden_dim = n_heads * channels_per_head
        self.norm = GroupNorm(channels)

        self.proj_in = ConvND(dim)(channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        blocks = [BasicTransformerBlock(channels, n_heads, p_dropout=dropout, cond_dim=cond_dim) for _ in range(depth)]
        self.transformer_blocks = Sequential(*blocks)

        self.proj_out = zero_init(ConvND(dim)(hidden_dim, channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, cond: torch.Tensor | None = None):
        skip = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = self.transformer_blocks(x, cond=cond)
        x = self.proj_out(x)
        return x + skip
