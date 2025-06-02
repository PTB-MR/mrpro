"""Uformer: U-Net with window attention."""

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch.nn import GELU, LeakyReLU, Module

from mrpro.nn.DropPath import DropPath
from mrpro.nn.FiLM import FiLM
from mrpro.nn.join import Concat
from mrpro.nn.ndmodules import ConvND, ConvTransposeND, InstanceNormND
from mrpro.nn.nets.UNet import UNetBase, UNetDecoder, UNetEncoder
from mrpro.nn.Sequential import Sequential
from mrpro.nn.ShiftedWindowAttention import ShiftedWindowAttention


class LeWinTransformerBlock(Module):
    """Locally-enhanced windowed attention transformer block.

    Part of the Uformer architecture.
    """

    def __init__(
        self,
        dim: int,
        n_channels_per_head: int,
        n_heads: int,
        window_size: int = 8,
        shifted: bool = False,
        mlp_ratio: float = 4.0,
        p_droppath: float = 0.0,
        cond_dim: int = 0,
    ) -> None:
        """Initialize the LeWinTransformerBlock module.

        Parameters
        ----------
        dim : int
            Dimension of the input, e.g. 2 or 3
        n_channels_per_head : int
            Number of features per head
        n_heads : int
            Number of attention heads
        window_size : int, optional
            Size of the attention window
        shifted : bool, optional
            Whether to use shifted variant of the attention
        mlp_ratio : float, optional
            Ratio of the hidden dimension to the input dimension
        p_droppath : float, optional
            Dropout probability for the drop path.
        cond_dim : int, optional
            Dimension of a conditioning tensor. If `0`, no FiLM layers are added.
        """
        super().__init__()
        channels = n_channels_per_head * n_heads
        hidden_dim = int(channels * mlp_ratio)
        self.norm1 = InstanceNormND(dim)(channels)
        self.attn = ShiftedWindowAttention(
            dim=dim,
            channels_in=channels,
            channels_out=channels,
            n_heads=n_heads,
            window_size=window_size,
            shifted=shifted,
        )
        self.norm2 = InstanceNormND(dim)(channels)
        self.ff = Sequential(
            ConvND(dim)(channels, hidden_dim, 1),
            GELU(),
            ConvND(dim)(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, stride=1, padding=1),
            GELU(),
            ConvND(dim)(hidden_dim, channels, 1),
        )
        if cond_dim > 0:
            self.ff.append(FiLM(channels, cond_dim))
        self.modulator = torch.nn.Parameter(torch.empty(channels, *((window_size,) * dim)))
        torch.nn.init.trunc_normal_(self.modulator)
        self.drop_path = DropPath(droprate=p_droppath)

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the transformer block.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning tensor

        Returns
        -------
        Output tensor
        """
        return super().__call__(x, cond=cond)


class Uformer(UNetBase):
    """Uformer: U-Net with window attention.

    Implements the Uformer network proposed in [WANG21]_
    It is SWin-Transformer/U-Net hybrid consisting of (shifted) windows attention transformer layers at different
    resolution levels, extended by FiLM layers for conditioning.

    References
    ----------
    .. [WANG21] Wang, Z., Cun, X., Bao, J., Zhou, W., Liu, J., & Li, H. Uformer: A general u-shaped transformer for
       image restoration. CVPR 2022. https://doi.org/10.48550/arXiv.2106.03106
    """

    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        n_channels_per_head: int = 32,
        n_heads: Sequence[int] = (1, 2, 4, 8),
        n_blocks: int = 2,
        cond_dim: int = 0,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        max_droppath_rate: float = 0.1,
    ):
        """Initialize the Uformer module.

        Parameters
        ----------
        dim : int
            Dimension of the input, e.g. 2 or 3
        channels_in : int
            Number of input channels
        channels_out : int
            Number of output channels
        n_channels_per_head : int, optional
            Number of features per head. The number of features at a resolution level is given by
            `n_channels_per_head * n_heads`.
        n_heads : Sequence[int], optional
            Number of attention heads at each resolution level.
        n_blocks : int, optional
            Number of transformer blocks at each resolution level in the input and output path
        cond_dim : int, optional
            Dimension of a conditioning tensor. If `0`, no FiLM layers are added.
        window_size : int, optional
            Size of the attention windows in the (shifted) window attention layers.
        mlp_ratio : float, optional
            Ratio of the hidden dimension to the input dimension in the feed-forward blocks
        max_droppath_rate : float, optional
            Maximum drop path rate. As in the original implementation, the drop path rate in the input path
            is linearly increased from `0` to `max_droppath_rate` with decreasing resolution. The rate in output
            blocks is fixed to `max_droppath_rate`.
        """

        def blocks(n_heads: int, p_droppath: float = 0.0):
            return Sequential(
                *(
                    LeWinTransformerBlock(
                        dim=dim,
                        n_heads=n_heads,
                        n_channels_per_head=n_channels_per_head,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        shifted=bool(i % 2),
                        p_droppath=p_droppath,
                        cond_dim=cond_dim,
                    )
                    for i in range(n_blocks)
                )
            )

        first_block = torch.nn.Sequential(
            ConvND(dim)(channels_in, n_channels_per_head * n_heads[0], kernel_size=3, stride=1, padding='same'),
            LeakyReLU(),
        )
        drop_path_rates = torch.linspace(0, max_droppath_rate, len(n_heads)).tolist()
        encoder_blocks = [
            blocks(n_heads=n_head, p_droppath=p_droppath_input)
            for n_head, p_droppath_input in zip(n_heads[:-1], drop_path_rates[:-1], strict=True)
        ]
        down_blocks = [
            ConvND(dim)(
                n_channels_per_head * n_head_current,
                n_channels_per_head * n_head_next,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            for n_head_current, n_head_next in pairwise(n_heads)
        ]
        middle_block = blocks(n_heads=n_heads[-1], p_droppath=max_droppath_rate)
        encoder = UNetEncoder(
            first_block=first_block,
            blocks=encoder_blocks,
            down_blocks=down_blocks,
            middle_block=middle_block,
        )

        decoder_blocks = [blocks(n_heads=2 * n_head, p_droppath=max_droppath_rate) for n_head in reversed(n_heads[:-1])]
        concat_blocks = [Concat() for _ in range(len(decoder_blocks))]
        up_blocks = [
            ConvTransposeND(dim)(
                n_channels_per_head * n_heads[-1], n_channels_per_head * n_heads[-2], kernel_size=2, stride=2
            )
        ]
        for n_head_current, n_head_next in pairwise(reversed(n_heads[:-1])):
            up_blocks.append(
                ConvTransposeND(dim)(
                    2 * n_channels_per_head * n_head_current, n_channels_per_head * n_head_next, kernel_size=2, stride=2
                )
            )
        last_block = ConvND(dim)(
            2 * n_channels_per_head * n_heads[0], channels_out, kernel_size=3, stride=1, padding='same'
        )
        decoder = UNetDecoder(
            blocks=decoder_blocks,
            concat_blocks=concat_blocks,
            up_blocks=up_blocks,
            last_block=last_block,
        )

        super().__init__(encoder=encoder, decoder=decoder)
