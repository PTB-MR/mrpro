"""Gateded MBConv Residual Block."""

import torch
from torch.nn import Identity, Module, Sequential, SiLU

from mrpro.nn.CondMixin import CondMixin
from mrpro.nn.FiLM import FiLM
from mrpro.nn.ndmodules import convND
from mrpro.nn.RMSNorm import RMSNorm


class GluMBConvResBlock(CondMixin, Module):
    """Gated MBConv residual block.

    Gated variant [DCAE]_ of the MBConv block [EffNet]_ with a residual connection and (optional) conditioning.

    References
    ----------
    .. [DCAE] Chen et al. Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models. ICLR 2025
       https://arxiv.org/abs/2410.10733
    .. [EffNet] Tan et al. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019
       https://arxiv.org/abs/1905.11946
    """

    def __init__(
        self,
        n_dim: int,
        n_channels_in: int,
        n_channels_out: int,
        expand_ratio: int = 6,
        stride: int = 1,
        kernel_size: int = 3,
        cond_dim: int = 0,
    ):
        """Initialize MBConv block.

        Parameters
        ----------
        n_dim
            Number of spatial dimensions.
        n_channels_in
            Number of input channels.
        n_channels_out
            Number of output channels.
        expand_ratio
            Expansion ratio inside the block.
        stride
            Stride of the depthwise convolution.
        kernel_size
            Kernel size of the depthwise convolution.
        cond_dim
            Dimension of the conditioning tensor used in a FiLM. If 0, no FiLM is used.
        """
        super().__init__()
        channels_mid = n_channels_in * expand_ratio
        if stride == 1 and n_channels_in == n_channels_out:
            self.skip: Module = Identity()
        else:
            self.skip = convND(n_dim)(n_channels_in, n_channels_out, kernel_size=1, stride=stride)
        self.inverted_conv = Sequential(
            convND(n_dim)(
                n_channels_in,
                channels_mid * 2,
                kernel_size=1,
            ),
            SiLU(),
        )
        self.depth_conv = Sequential(
            convND(n_dim)(
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
            convND(n_dim)(
                channels_mid,
                n_channels_out,
                kernel_size=1,
            ),
            RMSNorm(n_channels_out),
            SiLU(),
        )
        if cond_dim > 0:
            self.film: FiLM | None = FiLM(channels_mid, cond_dim)
        else:
            self.film = None

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply MBConv block.

        Parameters
        ----------
        x
            Input tensor.
        cond
            Conditioning tensor. If `None`, no conditioning is applied.

        Returns
        -------
            Output tensor.
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply MBConv block."""
        h = self.inverted_conv(x)
        h = self.depth_conv(h)
        h, gate = torch.chunk(h, 2, dim=1)
        h = h * torch.nn.functional.silu(gate)
        if self.film is not None:
            h = self.film(h, cond=cond)
        h = self.point_conv(h)
        return self.skip(x) + h
