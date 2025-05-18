import torch
from torch.nn import GELU, Module

from mrpro.nn.FiLM import FiLM
from mrpro.nn.LayerNorm import LayerNorm
from mrpro.nn.NDModules import ConvND
from mrpro.nn.Sequential import Sequential
from mrpro.nn.ShiftedWindowAttention import ShiftedWindowAttention


class SwinTransformerLayer(Module):
    """Swin Transformer layer.

    As used in the SwinIR network.
    """

    def __init__(
        self,
        dim: int,
        channels: int,
        n_heads: int,
        window_size: int,
        shifted: bool,
        mlp_ratio: int = 4,
        cond_dim: int = 0,
    ):
        """Initialize the Swin Transformer layer.

        Parameters
        ----------
        dim
            Number of spatial dimensions (1D, 2D, or 3D)
        channels
            Number of channels in the input tensor
        n_heads
            Number of attention heads
        window_size
            Size of the local window for computing windowed self-attention
        shifted
            Whether to use shifted window attention
        mlp_ratio
            Expansion ratio for the MLP
        cond_dim
            Dimension of optional tensor for FiLM conditioning. If 0, no conditioning is used
        """
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = ShiftedWindowAttention(dim, channels, n_heads, window_size, shifted)
        if cond_dim > 0:
            self.norm2 = Sequential(LayerNorm(None), FiLM(cond_dim))
        else:
            self.norm2 = Sequential(LayerNorm(channels))
        self.mlp = Sequential(
            ConvND(dim)(channels, channels * mlp_ratio, 1),
            GELU(),
            ConvND(dim)(channels * mlp_ratio, channels, 1),
        )

    def __call__(self, x, cond: torch.Tensor | None = None):
        """Apply the Swin Transformer layer.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, channels, *spatial_dims)
        cond
            Optional conditioning tensor of shape (batch_size, cond_dim)

        Returns
        -------
            Output tensor of shape (batch_size, channels, *spatial_dims)
        """
        return super().__call__(x, cond)

    def forward(self, x, cond: torch.Tensor | None = None):
        """Apply the Swin Transformer layer."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x, cond))
        return x


class RSTB(Module):
    """Residual Swin Transformer block.

    As used in the SwinIR network.
    """

    def __init__(self, dim: int, channels: int, n_heads: int, window_size: int, depth: int, cond_dim: int = 0):
        super().__init__()
        self.layers = Sequential(
            *[
                SwinTransformerLayer(dim, channels, n_heads, window_size, shifted=(i % 2 == 1), cond_dim=cond_dim)
                for i in range(depth)
            ]
        )
        self.conv = ConvND(dim)(channels, channels, 3, padding=1)

    def __call__(self, x, cond: torch.Tensor | None = None):
        """Apply the residual Swin Transformer block.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, channels, *spatial_dims)
        cond
            Optional conditioning tensor of shape (batch_size, cond_dim)

        Returns
        -------
            Output tensor of shape (batch_size, channels, *spatial_dims)
        """
        return super().__call__(x, cond)

    def forward(self, x, cond: torch.Tensor | None = None):
        """Apply the residual Swin Transformer block."""
        return x + self.conv(self.layers(x, cond))


class SwinIR(Module):
    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        channels_per_head: int = 16,
        n_heads: int = 6,
        window_size: int = 64,
        n_blocks: int = 6,
        n_attn_per_block: int = 6,
        cond_dim: int = 0,
    ):
        """Initialize the SwinIR model.

        Parameters
        ----------
        dim
            Number of spatial dimensions (1D, 2D, or 3D)
        channels_in
            Number of input channels
        channels_out
            Number of output channels
        channels_per_head
            Number of channels per attention head
        n_heads
            Number of attention heads
        window_size
            Size of the local window for computing windowed self-attention
        n_blocks
            Number of residual Swin Transformer blocks (RSTB)
        n_attn_per_block
            Number of windowed attention layers per RSTB block
        cond_dim
            Dimension of optional tensor for FiLM conditioning. If 0, no conditioning is used
        """
        super().__init__()
        self.shallow = ConvND(dim)(channels_in, channels_per_head * n_heads, 3, padding=1)
        self.body = Sequential(
            *[
                RSTB(dim, channels_per_head * n_heads, n_heads, window_size, n_attn_per_block, cond_dim)
                for _ in range(n_blocks)
            ]
        )
        self.body.append(ConvND(dim)(channels_per_head * n_heads, channels_per_head * n_heads, 3, padding=1))
        self.final = ConvND(dim)(channels_per_head, channels_out, 3, padding=1)
        self.skip = ConvND(dim)(channels_in, channels_out, 1, padding=1)

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the SwinIR model.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, channels_in, *spatial_dims)
        cond
            Optional conditioning tensor of shape (batch_size, cond_dim)

        Returns
        -------
        out
            Output tensor of shape (batch_size, channels_out, *spatial_dims)
        """
        return super().__call__(x, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the SwinIR model."""
        h = self.shallow(x)
        h = self.body(h, cond) + self.skip(x)
        out = self.final(h)
        return out
