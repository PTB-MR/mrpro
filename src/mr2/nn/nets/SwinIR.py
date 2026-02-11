"""SwinIR implementation."""

import torch
from torch.nn import GELU, Module

from mr2.nn.attention.ShiftedWindowAttention import ShiftedWindowAttention
from mr2.nn.DropPath import DropPath
from mr2.nn.FiLM import FiLM
from mr2.nn.ndmodules import convND, instanceNormND
from mr2.nn.Sequential import Sequential


class SwinTransformerLayer(Module):
    """Swin Transformer layer.

    Implements a single layer of the Swin Transformer architecture.
    """

    def __init__(
        self,
        n_dim: int,
        n_channels: int,
        n_heads: int,
        window_size: int,
        mlp_ratio: int = 4,
        emb_dim: int = 0,
        p_droppath: float = 0.0,
    ):
        """Initialize SwinTransformerLayer.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels
            The number of channels in the input tensor.
        n_heads
            Number of attention heads
        window_size
            Size of the attention window
        mlp_ratio
            Ratio for hidden dimension expansion in MLP
        emb_dim
            Dimension of conditioning input. If 0, no FiLM conditioning is used.
        p_droppath
            Droppath probability for MLP
        """
        super().__init__()
        self.norm1 = instanceNormND(n_dim)(n_channels)
        self.attn = ShiftedWindowAttention(n_dim, n_channels, n_channels, n_heads, window_size)
        self.norm2 = Sequential(instanceNormND(n_dim)(n_channels))
        if emb_dim > 0:
            self.norm2.append(FiLM(channels=n_channels, cond_dim=emb_dim))
        self.mlp = Sequential(
            convND(n_dim)(n_channels, n_channels * mlp_ratio, 1),
            GELU('tanh'),
            convND(n_dim)(n_channels * mlp_ratio, n_channels, 1),
            DropPath(p_droppath),
        )

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the Swin Transformer layer.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning input

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the Swin Transformer layer."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x, cond=cond))
        return x


class ResidualSwinTransformerBlock(Module):
    """Residual Swin Transformer block (RSTB).

    Combines a Swin Transformer layer with a residual connection,
    as used in the SwinIR architecture.
    """

    def __init__(
        self,
        n_dim: int,
        n_channels: int,
        n_heads: int,
        window_size: int,
        depth: int,
        emb_dim: int = 0,
        p_droppath: float = 0.0,
        mlp_ratio: int = 4,
    ):
        """Initialize ResidualSwinTransformerBlock.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels
            The number of channels in the input tensor.
        n_heads
            Number of attention heads
        window_size
            Size of the attention window
        depth
            Number of Swin Transformer layers
        emb_dim
            Dimension of conditioning input. If 0, no FiLM conditioning is used.
        p_droppath
            Droppath probability for MLP.
        mlp_ratio
            Ratio for hidden dimension expansion in MLP
        """
        super().__init__()
        self.layers = Sequential(
            *(
                SwinTransformerLayer(
                    n_dim, n_channels, n_heads, window_size, emb_dim=emb_dim, p_droppath=p_droppath, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            )
        )
        self.conv = convND(n_dim)(n_channels, n_channels, 3, padding=1)

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the residual Swin Transformer block.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning input. If None, no FiLM conditioning is used.

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the residual Swin Transformer block."""
        return x + self.conv(self.layers(x, cond=cond))


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
        n_dim: int,
        n_channels_in: int,
        n_channels_out: int,
        n_channels_per_head: int = 16,
        n_heads: int = 6,
        window_size: int = 64,
        n_blocks: int = 6,
        n_attn_per_block: int = 6,
        emb_dim: int = 0,
        p_droppath: float = 0.0,
        mlp_ratio: int = 4,
    ):
        """Initialize SwinIR.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels_in
            The number of input channels.
        n_channels_out
            The number of output channels.
        n_channels_per_head
            The number of channels per attention head.
        n_heads
            The number of attention heads.
        window_size
            The size of the attention window. Inputs sizes must be divisible by this value.
        n_blocks
            The number of residual blocks.
        n_attn_per_block
            The number of attention layers per block.
        emb_dim
            The dimension of the conditioning input. If 0, no FiLM conditioning is used.
        p_droppath
            The droppath probability for MLP.
        mlp_ratio
            The ratio for hidden dimension expansion in MLP.
        """
        super().__init__()
        self.first = convND(n_dim)(n_channels_in, n_channels_per_head * n_heads, kernel_size=3, padding=1)
        self.blocks = Sequential(
            *(
                ResidualSwinTransformerBlock(
                    n_dim,
                    n_channels_per_head * n_heads,
                    n_heads,
                    window_size,
                    n_attn_per_block,
                    emb_dim,
                    p_droppath,
                    mlp_ratio,
                )
                for _ in range(n_blocks)
            )
        )
        self.last = convND(n_dim)(n_channels_per_head * n_heads, n_channels_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply SwinIR.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning input. If None, no FiLM conditioning is used.

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = self.first(x)
        x = self.blocks(x, cond=cond)
        x = self.last(x)
        return x
