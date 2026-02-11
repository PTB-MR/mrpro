"""Spatial transformer block."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch.nn import Dropout, Linear, Module

from mr2.nn.attention.MultiHeadAttention import MultiHeadAttention
from mr2.nn.attention.NeighborhoodSelfAttention import NeighborhoodSelfAttention
from mr2.nn.CondMixin import CondMixin
from mr2.nn.GEGLU import GEGLU
from mr2.nn.GroupNorm import GroupNorm
from mr2.nn.LayerNorm import LayerNorm
from mr2.nn.PermutedBlock import PermutedBlock
from mr2.nn.RMSNorm import RMSNorm
from mr2.nn.Sequential import Sequential


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
        rope_embed_fraction: float = 0.0,
        attention_neighborhood: int | None = None,
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
        rope_embed_fraction
            Fraction of channels to embed with RoPE.
        attention_neighborhood
            If not None, use neighborhood self attention with the given neighborhood size instead
            of global self attention.
        """
        super().__init__()
        self.features_last = features_last

        if attention_neighborhood is None:
            attention: Module = MultiHeadAttention(
                n_channels_in=channels,
                n_channels_out=channels,
                n_heads=n_heads,
                p_dropout=p_dropout,
                features_last=True,
                rope_embed_fraction=rope_embed_fraction,
            )
        else:
            if p_dropout > 0:
                raise ValueError('p_dropout > 0 is not supported for neighborhood self attention')
            attention = NeighborhoodSelfAttention(
                n_channels_in=channels,
                n_channels_out=channels,
                n_heads=n_heads,
                features_last=True,
                kernel_size=attention_neighborhood,
                circular=True,
                rope_embed_fraction=rope_embed_fraction,
            )
        self.selfattention = Sequential(LayerNorm(channels, features_last=True, cond_dim=cond_dim), attention)
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
        x = self.selfattention(x, cond=cond) + x
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
        p_dropout: float = 0.0,
        cond_dim: int = 0,
        rope_embed_fraction: float = 0.0,
        attention_neighborhood: int | None = None,
        features_last: bool = False,
        norm: Literal['group', 'rms'] = 'group',
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
        p_dropout
            Dropout probability.
        cond_dim
            Dimension of the conditioning tensor.
        rope_embed_fraction
            Fraction of channels to embed with RoPE.
        attention_neighborhood
            If not None, use NeighborhoodSelfAttention with the given neighborhood size instead of MultiHeadAttention.
        features_last
            Whether the features are last in the input tensor, as common in transformer models.
        norm
            Whether to use GroupNorm or RMSNorm.
        """
        super().__init__()
        hidden_dim = n_heads * (channels // n_heads)
        match norm:
            case 'group':
                self.norm: Module = GroupNorm(channels, features_last=features_last)
            case 'rms':
                self.norm = RMSNorm(channels, features_last=features_last)
            case _:
                raise ValueError(f'Invalid norm: {norm}')
        self.features_last = features_last
        self.proj_in = Linear(channels, hidden_dim)
        self.transformer_blocks = Sequential()
        for group in (g for _ in range(depth) for g in dim_groups):
            if not self.features_last:
                group = tuple(g - 1 if g < 0 else g for g in group)
            block = BasicTransformerBlock(
                hidden_dim,
                n_heads,
                p_dropout=p_dropout,
                cond_dim=cond_dim,
                features_last=True,
                rope_embed_fraction=rope_embed_fraction,
                attention_neighborhood=attention_neighborhood,
            )
            self.transformer_blocks.append(PermutedBlock(group, block, features_last=True))
        self.proj_out = zero_init(Linear(hidden_dim, channels))

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the spatial transformer block."""
        skip = x
        h = self.norm(x)
        if not self.features_last:
            h = h.movedim(1, -1)
        h = self.proj_in(h)
        h = self.transformer_blocks(h, cond=cond)
        h = self.proj_out(h)
        if not self.features_last:
            h = h.movedim(-1, 1)
        return skip + h

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the spatial transformer block.

        Parameters
        ----------
        x
            Input tensor.
        cond
            Conditioning tensor. If None, no conditioning is applied.

        Returns
        -------
            Output tensor.
        """
        return super().__call__(x, cond=cond)
