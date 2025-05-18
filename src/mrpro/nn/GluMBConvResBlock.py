"""Gateded MBConv Residual Block."""

import torch
from torch.nn import Identity, Module, Sequential, SiLU

from mrpro.nn.EmbMixin import EmbMixin
from mrpro.nn.FiLM import FiLM
from mrpro.nn.NDModules import ConvND
from mrpro.nn.RMSNorm import RMSNorm


class GluMBConvResBlock(EmbMixin, Module):
    """Gated MBConv residual block.

    Gated variant [DCAE]_ of the MBConv block [EffNet]_ with a residual connection.

    References
    ----------
    .. [DCAE] Chen et al. Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models. ICLR 2025
       https://arxiv.org/abs/2410.10733
    .. [EffNet] Tan et al. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019
       https://arxiv.org/abs/1905.11946
    """

    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        expand_ratio: int = 6,
        stride: int = 1,
        kernel_size: int = 3,
        emb_dim: int = 0,
    ):
        """Initialize MBConv block.

        Parameters
        ----------
        dim
            Number of spatial dimensions.
        channels_in
            Number of input channels.
        channels_out
            Number of output channels.
        expand_ratio
            Expansion ratio inside the block.
        stride
            Stride of the depthwise convolution.
        kernel_size
            Kernel size of the depthwise convolution.
        emb_dim
            Size of the FiLM embedding. If 0, no embedding is used.
        """
        super().__init__()
        channels_mid = channels_in * expand_ratio
        if stride == 1 and channels_in == channels_out:
            self.skip: Module = Identity()
        else:
            self.skip = ConvND(dim)(channels_in, channels_out, kernel_size=1, stride=stride)
        self.inverted_conv = Sequential(
            ConvND(dim)(
                channels_in,
                channels_mid * 2,
                kernel_size=1,
            ),
            SiLU(),
        )
        self.depth_conv = Sequential(
            ConvND(dim)(
                channels_mid * 2,
                channels_mid * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding='same',
                groups=channels_mid * 2,
            ),
            SiLU(),
        )
        self.point_conv = Sequential(
            ConvND(dim)(
                channels_mid,
                channels_out,
                kernel_size=1,
            ),
            RMSNorm(channels_out),
            SiLU(),
        )
        if emb_dim > 0:
            self.film: FiLM | None = FiLM(channels_mid, emb_dim)
        else:
            self.film = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Apply MBConv block."""
        h = self.inverted_conv(x)
        h = self.depth_conv(h)
        h, gate = torch.chunk(h, 2, dim=1)
        h = h * torch.nn.functional.silu(gate)
        if self.film is not None:
            h = self.film(h, emb)
        h = self.point_conv(h)
        return self.skip(x) + h
