from collections.abc import Sequence

from torch.nn import Module

from mrpro.nn.attention.SpatialTransformerBlock import SpatialTransformerBlock
from mrpro.nn.join import Interpolate
from mrpro.nn.nets.UNet import UNetBase, UNetDecoder, UNetEncoder
from mrpro.nn.PixelShuffle import PixelShuffleUpsample, PixelUnshuffleDownsample
from mrpro.nn.Sequential import Sequential
from mrpro.operators.RearrangeOp import RearrangeOp
from mrpro.utils import to_tuple


class HourglassTransformer(UNetBase):
    def __init__(
        self,
        n_dim: int,
        n_channels_in: int,
        n_channels_out: int,
        n_features: Sequence[int] | int,
        depths: Sequence[int] | int = 3,
        attention_neighborhood: Sequence[None | int] | int | None = 11,
        n_heads: int | Sequence[int] = 4,
        cond_dim: int = 0,
    ):
        n_layers_ = [
            len(x)
            for x in (n_features, depths, attention_neighborhood, n_heads)
            if (x is not None and not isinstance(x, int))
        ]
        n_layers = n_layers_[0]

        if any(x != n_layers_[0] for x in n_layers_):
            raise ValueError('All arguments must have the same length or be scalars')

        n_features_ = to_tuple(n_layers, n_features)
        depths_ = to_tuple(n_layers, depths)
        attention_neighborhood_ = to_tuple(n_layers, attention_neighborhood)
        n_heads_ = to_tuple(n_layers, n_heads)

        move_channels_last = RearrangeOp('batch  ... channels -> batch ... channels')
        first_block = Sequential(
            PixelUnshuffleDownsample(n_dim, n_channels_in, n_features_[0], downscale_factor=2),
            move_channels_last,
        )
        dim = (tuple(range(-n_dim, 0)),)  # TODO: allow arbitrary dimensions.
        encoder_blocks: list[Module] = []
        decoder_blocks: list[Module] = []
        merge_blocks: list[Module] = []
        down_blocks: list[Module] = []
        up_blocks: list[Module] = []
        for channels, depth, neighborhood, head in zip(
            n_features_, depths_, attention_neighborhood_, n_heads_, strict=True
        ):
            encoder_blocks.append(
                SpatialTransformerBlock(
                    dim_groups=dim,
                    channels=channels,
                    depth=depth,
                    attention_neighborhood=neighborhood,
                    n_heads=head,
                    rope_embed_fraction=1.0,
                    cond_dim=cond_dim,
                )
            )
            decoder_blocks.append(
                SpatialTransformerBlock(
                    dim_groups=dim,
                    channels=channels,
                    depth=depth,
                    attention_neighborhood=neighborhood,
                    n_heads=head,
                    rope_embed_fraction=1.0,
                    cond_dim=cond_dim,
                )
            )
            merge_blocks.append(Interpolate())

        last_block = Sequential(
            move_channels_last.H, PixelShuffleUpsample(n_dim, n_features_[-1], n_channels_out, upscale_factor=2)
        )
        middle_block = SpatialTransformerBlock(
            dim_groups=dim,
            channels=n_features_[-1],
            depth=depths_[-1],
            attention_neighborhood=attention_neighborhood_[-1],
            n_heads=n_heads_[-1],
            rope_embed_fraction=1.0,
            cond_dim=cond_dim,
        )
        encoder = UNetEncoder(first_block, encoder_blocks, down_blocks, middle_block)
        decoder = UNetDecoder(decoder_blocks, up_blocks, merge_blocks, last_block)

        super().__init__(encoder, decoder)
