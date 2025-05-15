"""Restormer implementation."""

from collections.abc import Sequence
import torch
from torch.nn import Module, PixelUnshuffle, PixelShuffle
from mrpro.nn.TransposedAttention import TransposedAttention
from mrpro.nn.NDModules import ConvNd, InstanceNormNd
from mrpro.nn.FiLM import FiLM


class GDFN(Module):
    """Gated depthwise feed forward network.

    As used in the Restormer architecture.
    """

    def __init__(self, dim: int, channels: int, mlp_ratio: float):
        super().__init__()

        hidden_features = int(channels * mlp_ratio)
        self.project_in = ConvNd(dim)(channels, hidden_features * 2, kernel_size=1)
        self.depthwise_conv = ConvNd(dim)(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
        )
        self.project_out = ConvNd(dim)(hidden_features, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.depthwise_conv(x).chunk(2, dim=1)
        return self.project_out(torch.nn.functional.gelu(x1) * x2)


class RestormerBlock(Module):
    """Transformer block with transposed attention and gated depthwise feed forward network."""

    def __init__(self, dim: int, channels: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = InstanceNormNd(dim)(channels)
        self.attn = TransposedAttention(dim, channels, num_heads)
        self.norm2 = InstanceNormNd(dim)(channels)
        self.ffn = GDFN(dim, channels, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body =

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body =

    def forward(self, x):
        return self.body(x)


class Restormer(UNetBase):
    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        n_blocks: Sequence[int] = (4, 6, 6, 8),
        n_refinement_blocks: int = 4,
        n_heads: Sequence[int] = (1, 2, 4, 8),
        n_channels_per_head: int = 48,
        mlp_ratio: float = 2.66,
        emb_dim: int = 0,
    ):
        super().__init__()

        self.first = ConvNd(dim)(channels_in, n_channels_per_head, kernel_size=3, stride=1, padding=1, bias=False)

        def blocks(n_heads: int, n_blocks: int):
            layers = Sequential(
                *(RestormerBlock(dim, n_channels_per_head, n_heads, mlp_ratio) for i in range(n_blocks))
            )

            if emb_dim > 0 and n_blocks > 1:
                layers.insert(1, FiLM(channels=n_features_per_head * n_heads, channels_emb=emb_dim))
            return layers



        for n_block, n_heads in zip(n_blocks, n_heads):
            self.input_blocks.append(blocks(n_heads, n_block))
            self.output_blocks.append(blocks(n_heads, n_block))
            self.skip_blocks.append(Identity())


         for n_head_current, n_head_next in pairwise(n_heads):
            self.down_blocks.append(
               Sequential(
            ConvND(dim)(n_channels_per_head * n_head_current, n_channels_per_head * n_head_next, kernel_size=4, stride=2, padding=1,
                PixelUnshuffle(2)
            )
            self.up_blocks.append(
               nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False), PixelShuffle(2)
        )
            )
        self.output_blocks = self.output_blocks[::-1]
        self.middle_block = blocks(n_heads, n_blocks)

                    num_heads=heads[0],
                    ffn_expansion_factor=mlp_ratio,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
