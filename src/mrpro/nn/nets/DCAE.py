"""Deep Compression Autoencoder."""

from collections.abc import Sequence

import torch
from torch.nn import Module, Sequential, SiLU

from mrpro.nn.GluMBConvResBlock import GluMBConvResBlock
from mrpro.nn.LinearSelfAttention import LinearSelfAttention
from mrpro.nn.MultiHeadAttention import MultiHeadAttention
from mrpro.nn.NDModules import ConvND
from mrpro.nn.PixelShuffle import PixelShuffleUpsample, PixelUnshuffleDownsample
from mrpro.nn.Residual import Residual
from mrpro.nn.RMSNorm import RMSNorm


class CNNBlock(Residual):
    """Block with two convolutions and normalization."""

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
            Residual(
                Sequential(
                    ConvND(dim)(channels, channels, kernel_size=3, padding=1),
                    SiLU(),
                    ConvND(dim)(channels, channels, kernel_size=3, padding=1, bias=False),
                    RMSNorm(channels),
                )
            )
        )


class EfficientViTBlock(Module):
    """Efficient Vision Transformer block with optional linear attention."""

    def __init__(
        self,
        dim: int,
        channels: int,
        n_heads: int,
        expand_ratio: int = 4,
        linear_attn: bool = False,
    ):
        super().__init__()
        if linear_attn:
            attention: Module = LinearSelfAttention(channels, channels, n_heads)  # TODO: check heads and head dim
        else:
            attention = MultiHeadAttention(channels, channels, n_heads, features_last=False)
        self.context_module = Residual(Sequential(attention, RMSNorm(channels)))
        self.local_module = GluMBConvResBlock(
            dim=dim,
            channels_in=channels,
            channels_out=channels,
            expand_ratio=expand_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for EfficientViTBlock."""
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class Encoder(Sequential):
    """Encoder for DCAE."""

    def __init__(
        self,
        dim: int = 2,
        channels_in: int = 3,
        channels_out: int = 32,
        block_types: Sequence[str] = ('CNN', 'CNN', 'LinearViT', 'LinearViT', 'ViT'),
        widths: Sequence[int] = (256, 512, 512, 1024, 1024),
        depths: Sequence[int] = (4, 6, 2, 2, 2),
    ):
        super().__init__()
        self.append(PixelUnshuffleDownsample(dim, channels_in, widths[0], downscale_factor=2, residual=False))
        if len(block_types) != len(widths) or len(block_types) != len(depths):
            raise ValueError('block_types, widths, and depths must have the same length')
        for block_type, width, depth in zip(block_types, widths, depths, strict=False):
            match block_type:
                case 'CNN':
                    stage: list[Module] = [CNNBlock(dim, width) for _ in range(depth)]
                case 'LinearViT':
                    stage = [
                        EfficientViTBlock(dim, width, n_heads=1, linear_attn=True) for _ in range(depth)
                    ]  # TODO: heads
                case 'ViT':
                    stage = [EfficientViTBlock(dim, width, n_heads=1, linear_attn=False) for _ in range(depth)]
                case _:
                    raise ValueError(f'Block type {block_type} not supported')
            self.append(Sequential(*stage))
            if len(self) < len(widths):
                self.append(PixelUnshuffleDownsample(dim, width, width, downscale_factor=2, residual=True))
        self.append(PixelUnshuffleDownsample(dim, widths[-1], channels_out, downscale_factor=1, residual=True))


class Decoder(Sequential):
    """Decoder for DCAE."""

    def __init__(
        self,
        dim: int = 2,
        channels_in: int = 32,
        channels_out: int = 3,
        block_types: Sequence[str] = ('ViT', 'LinearViT', 'LinearViT', 'CNN', 'CNN'),
        widths: Sequence[int] = (1024, 1024, 512, 512, 256),
        depths: Sequence[int] = (2, 2, 2, 6, 4),
    ):
        super().__init__()
        if not (len(block_types) == len(widths) == len(depths)):
            raise ValueError('block_types, widths, and depths must have the same length')
        #  "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
        #     "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,2,2,2] "
        #     "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu]"
        self.append(PixelShuffleUpsample(dim, channels_in, widths[0], upscale_factor=1, residual=True))

        self.stages: list[Sequential] = []
        for block_type, width, depth in zip(block_types, widths, depths, strict=False):
            match block_type:
                case 'ResBlock':
                    stage: list[Module] = [CNNBlock(dim, width) for _ in range(depth)]
                case 'LinearViT':
                    stage = [
                        EfficientViTBlock(dim, width, n_heads=1, linear_attn=True) for _ in range(depth)
                    ]  # TODO: heads
                case 'ViT':
                    stage = [EfficientViTBlock(dim, width, n_heads=1, linear_attn=False) for _ in range(depth)]
                case _:
                    raise ValueError(f'Block type {block_type} not supported')

            self.stages.append(Sequential(*stage))
            if len(self) < len(widths):
                self.append(PixelShuffleUpsample(dim, width, width, upscale_factor=2, residual=True))

        #     stage.extend(
        #         build_stage_main(
        #             width=width,
        #             depth=depth,
        #             block_type=block_type,
        #             norm=norm,
        #             act=act,
        #             input_width=(
        #                 width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)]
        #             ),
        #         )
        #     )
        #     self.stages.insert(0, OpSequential(stage))
        # self.stages = nn.ModuleList(self.stages)

        # self.project_out = build_decoder_project_out_block(
        #     in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
        #     out_channels=cfg.in_channels,
        #     factor=1 if cfg.depth_list[0] > 0 else 2,
        #     upsample_block_type=cfg.upsample_block_type,
        #     norm=cfg.out_norm,
        #     act=cfg.out_act,
        # )

        self.project_out = PixelShuffleUpsample(dim, widths[-1], channels_out, upscale_factor=1, residual=True)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Forward pass for Decoder."""
    #     x = self.project_in(x)
    #     for stage in reversed(self.stages):
    #         x = stage(x)
    #     x = self.project_out(x)
    #     return x
