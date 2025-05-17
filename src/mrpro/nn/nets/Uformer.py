from collections.abc import Sequence
from itertools import pairwise

import torch
from sympy import Identity
from torch.nn import GELU, LeakyReLU, Module, Sequential

from mrpro.nn.NDModules import ConvND, ConvTransposeND, InstanceNormND
from mrpro.nn.nets.UNet import UNetBase
from mrpro.nn.ShiftedWindowAttention import ShiftedWindowAttention
from mrpro.nn.FiLM import FiLM
from mrpro.nn.Sequential import Sequential
from mrpro.nn.DropPath import DropPath


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
    """Locally-enhanced windowed attention transformer block.

    Part of the Uformer architecture.
    """

    def __init__(
        self,
        dim: int,
        n_features_per_head: int,
        n_heads: int,
        window_size: int = 8,
        shifted: bool = False,
        mlp_ratio: float = 4.0,
        p_droppath: float = 0.0,
    ) -> None:
        """Initialize the LeWinTransformerBlock module.

        Parameters
        ----------
        dim : int
            Dimension of the input, e.g. 2 or 3
        n_features_per_head : int
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
        """
        super().__init__()
        channels = n_features_per_head * n_heads
        self.norm1 = InstanceNormND(dim)(channels)
        self.attn = ShiftedWindowAttention(
            dim=dim,
            n_channels_per_head=n_features_per_head,
            n_heads=n_heads,
            window_size=window_size,
            shifted=shifted,
        )

        self.norm2 = InstanceNormND(dim)(channels)
        self.ff = LeFF(dim=dim, channels_in=channels, channels_out=channels, expand_ratio=mlp_ratio)
        self.modulator = torch.nn.Parameter(torch.empty(channels, *((window_size,) * dim)))
        torch.nn.init.trunc_normal_(self.modulator)
        self.drop_path = DropPath(droprate=p_droppath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformer block."""
        modulator = self.modulator.tile([t // s for t, s in zip(x.shape[1:], self.modulator.shape, strict=False)])
        x_mod = self.norm1(x) + modulator
        x_attn = self.attn(x_mod)
        x_ff = self.ff(self.norm2(x_attn))
        return x + self.drop_path(x_ff)


class Uformer(UNetBase):
    """Uformer: U-Net with window attention.

    Implements the Uformer network proposed in [WANG21]_
    It is SWIN/U-Net hybrid consisting of (shifted) windows attention transformer layers at different
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
        n_features_per_head: int = 32,
        n_heads: Sequence[int] = (1, 2, 4, 8),
        n_blocks: int = 2,
        emb_dim: int = 0,
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
        n_features_per_head : int, optional
            Number of features per head. The number of features at a resolution level is given by
            `n_features_per_head * n_heads`.
        n_heads : Sequence[int], optional
            Number of attention heads at each resolution level.
        n_blocks : int, optional
            Number of transformer blocks at each resolution level in the input and output path
        emb_dim : int, optional
            Dimension of the embedding. If `0`, no FiLM layers are added.
        window_size : int, optional
            Size of the attention windows in the (shifted) window attention layers.
        mlp_ratio : float, optional
            Ratio of the hidden dimension to the input dimension in the feed-forward blocks
        max_droppath_rate : float, optional
            Maximum drop path rate. As in the original implementation, the drop path rate in the input path
            is linearly increased from `0` to `max_droppath_rate` with decreasing resolution. The rate in output
            blocks is fixed to `max_droppath_rate`.
        """
        super().__init__()

        def blocks(n_heads: int, p_droppath: float = 0.0):
            layers = Sequential(
                *(
                    LeWinTransformerBlock(
                        dim=dim,
                        n_heads=n_heads,
                        n_features_per_head=n_features_per_head,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        shifted=bool(i % 2),
                        p_droppath=p_droppath,
                    )
                    for i in range(n_blocks)
                )
            )

            if emb_dim > 0 and n_blocks > 1:
                layers.insert(1, FiLM(channels=n_features_per_head * n_heads, channels_emb=emb_dim))
            return layers

        drop_path_rates = torch.linspace(0, max_droppath_rate, len(n_heads)).tolist()
        for n_head, p_droppath_input in zip(n_heads, drop_path_rates, strict=True):
            self.input_blocks.append(blocks(n_heads=n_head, p_droppath=p_droppath_input))
            self.output_blocks.append(blocks(n_heads=n_head, p_droppath=max_droppath_rate))
            self.skip_blocks.append(Identity())
        self.output_blocks = self.output_blocks[::-1]
        self.middle_block = blocks(n_heads=n_heads[-1], p_droppath=max_droppath_rate)

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
