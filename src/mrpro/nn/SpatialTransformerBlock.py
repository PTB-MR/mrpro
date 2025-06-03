"""Spatial transformer block."""

import torch
from torch.nn import Dropout, Linear, Module

from mrpro.nn.CondMixin import CondMixin
from mrpro.nn.GEGLU import GEGLU
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.LayerNorm import LayerNorm
from mrpro.nn.MultiHeadAttention import MultiHeadAttention
from mrpro.nn.ndmodules import ConvND
from mrpro.nn.Sequential import Sequential
from mrpro.nn.CondMixin import CondMixin


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
                channels_in=channels,
                channels_out=channels,
                n_heads=n_heads,
                p_dropout=p_dropout,
                features_last=True,
            ),
        )
        hidden_dim = int(channels * mlp_ratio)
        self.ff = Sequential(
            LayerNorm(channels, features_last=True),
            GEGLU(channels, hidden_dim, features_last=True),
            Dropout(p_dropout),
            Linear(hidden_dim, channels),
        )
        self.crossattention = (
            Sequential(
                LayerNorm(channels, features_last=True),
                MultiHeadAttention(
                    channels_in=channels,
                    channels_out=channels,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    channels_cross=cond_dim,
                    features_last=True,
                ),
            )
            if cond_dim > 0
            else None
        )
        self.cond_dim = cond_dim

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
            x = x.moveaxis(1, -1)
        x = self.selfattention(x) + x
        if cond is not None and self.crossattention is not None:
            cond = cond.unflatten(-1, (-1, self.cond_dim))
            x = self.crossattention(x, cond=cond) + x
        x = self.ff(x) + x
        if not self.features_last:
            x = x.moveaxis(-1, 1)
        return x


class SpatialTransformerBlock(CondMixin, Module):
    """Spatial transformer block."""

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
        """Initialize the spatial transformer block.

        Parameters
        ----------
        dim
            Spatial dimension of the input tensor.
        channels
            Number of channels in the input and output.
        n_heads
            Number of attention heads.
        channels_per_head
            Number of channels per attention head.
        depth
            Number of transformer blocks.
        dropout
            Dropout probability.
        cond_dim
            Number of channels in the conditioning tensor. If 0, no conditioning is applied.
        """
        super().__init__()
        self.in_channels = channels
        hidden_dim = n_heads * channels_per_head
        self.norm = GroupNorm(channels)

        self.proj_in = ConvND(dim)(channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        blocks = [BasicTransformerBlock(channels, n_heads, p_dropout=dropout, cond_dim=cond_dim) for _ in range(depth)]
        self.transformer_blocks = Sequential(*blocks)

        self.proj_out = zero_init(ConvND(dim)(hidden_dim, channels, kernel_size=1, stride=1, padding=0))

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the spatial transformer block.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning tensor. If None, no conditioning is applied.

        Returns
        -------
            Output tensor after spatial transformer
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the spatial transformer block."""
        skip = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = self.transformer_blocks(x, cond=cond)
        x = self.proj_out(x)
        return x + skip
