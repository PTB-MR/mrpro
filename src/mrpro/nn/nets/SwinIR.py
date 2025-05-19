"""SwinIR implementation."""

import torch
from torch.nn import Module

from mrpro.nn.FiLM import FiLM
from mrpro.nn.NDModules import ConvND, InstanceNormND
from mrpro.nn.Sequential import Sequential
from mrpro.nn.ShiftedWindowAttention import ShiftedWindowAttention


class SwinTransformerLayer(Module):
    """Swin Transformer layer.

    Implements a single layer of the Swin Transformer architecture.
    """

    def __init__(
        self,
        dim: int,
        channels: int,
        n_heads: int,
        window_size: int,
        mlp_ratio: int = 4,
        emb_dim: int = 0,
    ):
        """Initialize SwinTransformerLayer.

        Parameters
        ----------
        dim : int
            Dimension of the input space
        channels : int
            Number of input/output channels
        n_heads : int
            Number of attention heads
        window_size : int
            Size of the attention window
        mlp_ratio : int, optional
            Ratio for hidden dimension expansion, by default 4
        emb_dim : int, optional
            Dimension of conditioning input, by default 0
        """
        super().__init__()
        self.norm1 = Sequential(InstanceNormND(dim)(channels))
        self.attn = ShiftedWindowAttention(dim, channels, n_heads, window_size)
        self.norm2 = Sequential(InstanceNormND(dim)(channels))
        if emb_dim > 0:
            self.norm2.append(FiLM(channels=channels, cond_dim=emb_dim))

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the Swin Transformer layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        cond : torch.Tensor | None, optional
            Conditioning input, by default None

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return super().__call__(x, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the Swin Transformer layer."""
        x = x + self.attn(self.norm1(x))
        x = x + self.norm2(x)
        return x


class ResidualSwinTransformerBlock(Module):
    """Residual Swin Transformer block.

    Combines a Swin Transformer layer with a residual connection.
    """

    def __init__(
        self,
        dim: int,
        channels: int,
        n_heads: int,
        window_size: int,
        depth: int,
        emb_dim: int = 0,
    ):
        """Initialize ResidualSwinTransformerBlock.

        Parameters
        ----------
        dim : int
            Dimension of the input space
        channels : int
            Number of input/output channels
        n_heads : int
            Number of attention heads
        window_size : int
            Size of the attention window
        depth : int
            Number of Swin Transformer layers
        emb_dim : int, optional
            Dimension of conditioning input, by default 0
        """
        super().__init__()
        self.layers = Sequential(
            *(SwinTransformerLayer(dim, channels, n_heads, window_size, emb_dim=emb_dim) for _ in range(depth))
        )
        self.conv = ConvND(dim)(channels, channels, 3, padding=1)

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the residual Swin Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        cond : torch.Tensor | None, optional
            Conditioning input, by default None

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return super().__call__(x, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the residual Swin Transformer block."""
        return x + self.conv(self.layers(x, cond))


class SwinIR(Module):
    """SwinIR architecture.

    Implements the SwinIR [LZL21]_ network, which is a Swin Transformer based
    image restoration network.

    References
    ----------
    .. [LZL21] Liang, Jie, et al. "SwinIR: Image restoration using swin transformer."
       ICCVW 2021, https://arxiv.org/pdf/2108.10257.pdf
    """

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
        emb_dim: int = 0,
    ):
        """Initialize SwinIR.

        Parameters
        ----------
        dim : int
            Dimension of the input space
        channels_in : int
            Number of input channels
        channels_out : int
            Number of output channels
        channels_per_head : int, optional
            Number of channels per attention head, by default 16
        n_heads : int, optional
            Number of attention heads, by default 6
        window_size : int, optional
            Size of the attention window, by default 64
        n_blocks : int, optional
            Number of residual blocks, by default 6
        n_attn_per_block : int, optional
            Number of attention layers per block, by default 6
        emb_dim : int, optional
            Dimension of conditioning input, by default 0
        """
        super().__init__()
        self.first = ConvND(dim)(channels_in, channels_per_head * n_heads, kernel_size=3, padding=1)
        self.blocks = Sequential(
            *(
                ResidualSwinTransformerBlock(
                    dim,
                    channels_per_head * n_heads,
                    n_heads,
                    window_size,
                    n_attn_per_block,
                    emb_dim,
                )
                for _ in range(n_blocks)
            )
        )
        self.last = ConvND(dim)(channels_per_head * n_heads, channels_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply SwinIR.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        cond : torch.Tensor | None, optional
            Conditioning input, by default None

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = self.first(x)
        x = self.blocks(x, cond)
        x = self.last(x)
        return x
