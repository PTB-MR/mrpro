"""Spatial transformer block."""

from collections.abc import Sequence

import torch
from torch.nn import Dropout, Linear, Module

from mrpro.nn.attention.MultiHeadAttention import MultiHeadAttention
from mrpro.nn.CondMixin import CondMixin
from mrpro.nn.GEGLU import GEGLU
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.LayerNorm import LayerNorm
from mrpro.nn.PermutedBlock import PermutedBlock
from mrpro.nn.Sequential import Sequential


def zero_init(m: Module) -> Module:
    """Initialize module weights and bias to zero."""
    if hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
        torch.nn.init.zeros_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
        torch.nn.init.zeros_(m.bias)
    return m


class BasicTransformerBlock(CondMixin, Module):
    """Basic vision transformer block."""

    def __init__(
        self,
        channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        cond_dim: int = 0,
        mlp_ratio: float = 4,
        features_last: bool = False,
    ):
        """Initialize the basic transformer block.

        Parameters
        ----------
        channels
            Number of channels in the input and output.
        n_heads
            Number of attention heads.
        p_dropout
            Dropout probability.
        cond_dim
            Number of channels in the conditioning tensor.
        mlp_ratio
            Ratio of the hidden dimension to the input dimension.
        features_last
            Whether the features are last in the input tensor.
        """
        super().__init__()
        self.features_last = features_last
        self.selfattention = Sequential(
            LayerNorm(channels, features_last=True),
            MultiHeadAttention(
                n_channels_in=channels,
                n_channels_out=channels,
                n_heads=n_heads,
                p_dropout=p_dropout,
                features_last=True,
            ),
        )
        hidden_dim = int(channels * mlp_ratio)
        self.ff = Sequential(
            LayerNorm(channels, features_last=True, cond_dim=cond_dim),
            GEGLU(channels, hidden_dim, features_last=True),
            Dropout(p_dropout),
            Linear(hidden_dim, channels),
        )

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the basic transformer block.

        Parameters
        ----------
        x
            Input tensor.
        cond
            Conditioning tensor. If None, no conditioning is applied.
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the basic transformer block."""
        if not self.features_last:
            x = x.moveaxis(1, -1).contiguous()
        x = self.selfattention(x) + x
        x = self.ff(x, cond=cond) + x
        if not self.features_last:
            x = x.moveaxis(-1, 1).contiguous()
        return x


class SpatialTransformerBlock(CondMixin, Module):
    """Spatial transformer block."""

    def __init__(
        self,
        dim_groups: Sequence[tuple[int, ...]],
        channels: int,
        n_heads: int,
        depth: int = 1,
        dropout: float = 0.0,
        cond_dim: int = 0,
    ):
        """Initialize the spatial transformer block.

        Parameters
        ----------
        dim_groups
            Groups of spatial dimensions for separate attention mechanisms.
        channels
            Number of channels in the input and output.
        n_heads
            Number of attention heads for each group.
        depth
            Number of transformer blocks for each group.
        dropout
            Dropout probability.
        cond_dim
            Dimension of the conditioning tensor.
        """
        super().__init__()
        hidden_dim = n_heads * (channels // n_heads)
        self.norm = GroupNorm(channels)
        self.proj_in = Linear(channels, hidden_dim)
        self.transformer_blocks = Sequential()
        for group in (g for _ in range(depth) for g in dim_groups):
            group = tuple(g - 1 if g < 0 else g for g in group)
            block = BasicTransformerBlock(hidden_dim, n_heads, p_dropout=dropout, cond_dim=cond_dim, features_last=True)
            self.transformer_blocks.append(PermutedBlock(group, block, features_last=True))
        self.proj_out = Linear(hidden_dim, channels)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the spatial transformer block."""
        skip = x
        h = self.norm(x)
        h = h.movedim(1, -1)
        h = self.proj_in(h)
        h = self.transformer_blocks(h, cond=cond)
        h = self.proj_out(h)
        h = h.movedim(-1, 1)
        return skip + h

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        return super().__call__(x, cond=cond)
