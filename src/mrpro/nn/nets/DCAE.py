"""Deep Compression Autoencoder."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch.nn import Module, ReLU, SiLU

from mrpro.nn.GluMBConvResBlock import GluMBConvResBlock
from mrpro.nn.LinearSelfAttention import LinearSelfAttention
from mrpro.nn.MultiHeadAttention import MultiHeadAttention
from mrpro.nn.ndmodules import ConvND
from mrpro.nn.nets.VAE import VAE
from mrpro.nn.PixelShuffle import PixelShuffleUpsample, PixelUnshuffleDownsample
from mrpro.nn.Residual import Residual
from mrpro.nn.RMSNorm import RMSNorm
from mrpro.nn.Sequential import Sequential


class CNNBlock(Residual):
    """Block with two convolutions and normalization.

    As used in the DCAE [DCAE]_.

    References
    ----------
    .. [DCAE] Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., ... & Han, S. Deep compression autoencoder
       for efficient high-resolution diffusion models. ICLR 2025. https://arxiv.org/abs/2410.10733
    """

    def __init__(
        self,
        dim: int,
        channels: int,
    ):
        """Initialize the CNNBlock.

        Parameters
        ----------
        dim : int
            The spatial dimension of the input tensor.
        channels : int
            The number of channels in the input tensor.
        """
        super().__init__(
            Sequential(
                ConvND(dim)(channels, channels, kernel_size=3, padding=1),
                SiLU(True),
                ConvND(dim)(channels, channels, kernel_size=3, padding=1, bias=False),
                RMSNorm(channels),
            )
        )


class EfficientViTBlock(Module):
    """Efficient Vision Transformer block with optional linear attention.

    As used in the DCAE [DCAE]_.

    References
    ----------
    .. [DCAE] Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., ... & Han, S. Deep compression autoencoder
       for efficient high-resolution diffusion models. ICLR 2025. https://arxiv.org/abs/2410.10733
    """

    def __init__(
        self,
        dim: int,
        channels: int,
        n_heads: int,
        expand_ratio: int = 4,
        linear_attn: bool = False,
    ):
        """Initialize the EfficientViTBlock.

        Parameters
        ----------
        dim : int
            The spatial dimension of the input tensor.
        channels : int
            The number of channels in the input tensor.
        n_heads : int
            The number of attention heads.
        expand_ratio : int
            The expansion ratio of the GluMBConvResBlock.
        linear_attn : bool
            Whether to use linear attention instead of softmax attention with quadratic complexity.
        """
        super().__init__()
        if linear_attn:
            attention: Module = LinearSelfAttention(channels, channels, n_heads)
        else:
            attention = MultiHeadAttention(channels, channels, n_heads, features_last=False)
        self.context_module = Residual(Sequential(attention, RMSNorm(channels)))
        self.local_module = GluMBConvResBlock(
            n_dim=dim,
            n_channels_in=channels,
            n_channels_out=channels,
            expand_ratio=expand_ratio,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the EfficientViTBlock.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Output tensor
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for EfficientViTBlock."""
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class Encoder(Sequential):
    """Encoder for DCAE.

    As used in the DC-Autoencoder [DCAE]_.

    References
    ----------
    .. [DCAE] Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., ... & Han, S. Deep compression autoencoder
       for efficient high-resolution diffusion models. ICLR 2025. https://arxiv.org/abs/2410.10733
    """

    def __init__(
        self,
        dim: int = 2,
        channels_in: int = 3,
        channels_out: int = 32,
        block_types: Sequence[Literal['CNN', 'LinearViT', 'ViT']] = ('CNN', 'CNN', 'LinearViT', 'LinearViT', 'ViT'),
        widths: Sequence[int] = (256, 512, 512, 1024, 1024),
        depths: Sequence[int] = (4, 6, 2, 2, 2),
    ):
        """Initialize the Encoder.

        The length of the `block_types`, `widths`, and `depths` must be the same and determine
        the number of stages in the encoder. Between the stages, downsampling is performed.

        Parameters
        ----------
        dim : int
            The spatial dimension of the input tensor.
        channels_in : int
            The number of channels in the input tensor, i.e. the latent space
        channels_out : int
            The number of channels in the output tensor, i.e. the original space
        block_types : Sequence[str]
            The types of blocks to use in the decoder.
        widths : Sequence[int]
            The widths of the blocks in the decoder, i.e. the number of channels in the blocks
        depths : Sequence[int]
            The depths of the blocks in the decoder, i.e. the number blocks in the stage
        """
        super().__init__()
        self.append(PixelUnshuffleDownsample(dim, channels_in, widths[0], downscale_factor=2, residual=False))
        if len(block_types) != len(widths) or len(block_types) != len(depths):
            raise ValueError('block_types, widths, and depths must have the same length')
        for block_type, width, next_width, depth in zip(block_types, widths, (*widths[1:], None), depths, strict=False):
            match block_type:
                case 'CNN':
                    stage: list[Module] = [CNNBlock(dim, width) for _ in range(depth)]
                case 'LinearViT':
                    stage = [EfficientViTBlock(dim, width, max(1, width // 32), linear_attn=True) for _ in range(depth)]
                case 'ViT':
                    stage = [EfficientViTBlock(dim, width, max(1, width // 32)) for _ in range(depth)]
                case _:
                    raise ValueError(f'Block type {block_type} not supported')
            self.append(Sequential(*stage))
            if next_width:
                self.append(PixelUnshuffleDownsample(dim, width, next_width, downscale_factor=2, residual=True))
        self.append(
            Sequential(
                RMSNorm(widths[-1]),
                ReLU(),
                PixelUnshuffleDownsample(dim, widths[-1], channels_out, downscale_factor=1, residual=True),
            )
        )


class Decoder(Sequential):
    """Decoder for DCAE.

    As used in the DC-Autoencoder [DCAE]_.

    References
    ----------
    .. [DCAE] Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., ... & Han, S. Deep compression autoencoder
       for efficient high-resolution diffusion models. ICLR 2025. https://arxiv.org/abs/2410.10733
    """

    def __init__(
        self,
        dim: int = 2,
        channels_in: int = 32,
        channels_out: int = 3,
        block_types: Sequence[Literal['ViT', 'LinearViT', 'CNN']] = ('ViT', 'LinearViT', 'LinearViT', 'CNN', 'CNN'),
        widths: Sequence[int] = (1024, 1024, 512, 512, 256),
        depths: Sequence[int] = (2, 2, 2, 6, 4),
    ):
        """Initialize the Decoder.

        The length of the `block_types`, `widths`, and `depths` must be the same and determine
        the number of stages in the decoder. Between the stages, upsampling is performed.

        Parameters
        ----------
        dim : int
            The spatial dimension of the input tensor.
        channels_in : int
            The number of channels in the input tensor, i.e. the latent space
        channels_out : int
            The number of channels in the output tensor, i.e. the original space
        block_types : Sequence[str]
            The types of blocks to use in the decoder.
        widths : Sequence[int]
            The widths of the blocks in the decoder, i.e. the number of channels in the blocks
        depths : Sequence[int]
            The depths of the blocks in the decoder, i.e. the number blocks in the stage
        """
        super().__init__()
        if not (len(block_types) == len(widths) == len(depths)):
            raise ValueError('block_types, widths, and depths must have the same length')
        self.append(PixelShuffleUpsample(dim, channels_in, widths[0], upscale_factor=1, residual=True))

        for block_type, width, next_width, depth in zip(block_types, widths, (*widths[1:], None), depths, strict=False):
            match block_type:
                case 'CNN':
                    stage: list[Module] = [CNNBlock(dim, width) for _ in range(depth)]
                case 'LinearViT':
                    stage = [EfficientViTBlock(dim, width, n_heads=width // 32, linear_attn=True) for _ in range(depth)]
                case 'ViT':
                    stage = [
                        EfficientViTBlock(dim, width, n_heads=width // 32, linear_attn=False) for _ in range(depth)
                    ]
                case _:
                    raise ValueError(f'Block type {block_type} not supported')
            self.append(Sequential(*stage))
            if next_width:
                self.append(PixelShuffleUpsample(dim, width, next_width, upscale_factor=2, residual=True))

        self.append(
            Sequential(
                RMSNorm(widths[-1]),
                ReLU(),
                PixelShuffleUpsample(dim, widths[-1], channels_out, upscale_factor=2),
            )
        )


class DCVAE(VAE):
    """Variational Autoencoder based on DCAE.

    References
    ----------
    .. [DCAE] Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., ... & Han, S. Deep compression autoencoder
       for efficient high-resolution diffusion models. ICLR 2025. https://arxiv.org/abs/2410.10733
    """

    def __init__(
        self,
        dim: int,
        channels: int,
        latent_dim: int = 32,
        block_types: Sequence[Literal['CNN', 'LinearViT', 'ViT']] = ('CNN', 'CNN', 'LinearViT', 'LinearViT', 'ViT'),
        widths: Sequence[int] = (256, 512, 512, 1024, 1024),
        depths: Sequence[int] = (4, 6, 2, 2, 2),
    ):
        """Initialize the DCVAE."""
        encoder = Encoder(dim, channels, latent_dim * 2, block_types, widths, depths)
        decoder = Decoder(dim, latent_dim, channels, block_types[::-1], widths[::-1], depths[::-1])
        super().__init__(encoder, decoder)
