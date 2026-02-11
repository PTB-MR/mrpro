"""Hourglass Transformer."""

from collections.abc import Sequence
from itertools import pairwise

from torch.nn import Module

from mr2.nn.attention.SpatialTransformerBlock import SpatialTransformerBlock
from mr2.nn.join import Interpolate
from mr2.nn.nets.UNet import UNetBase, UNetDecoder, UNetEncoder
from mr2.nn.PixelShuffle import PixelShuffleUpsample, PixelUnshuffleDownsample
from mr2.nn.Sequential import Sequential
from mr2.operators.RearrangeOp import RearrangeOp
from mr2.utils.to_tuple import to_tuple


class HourglassTransformer(UNetBase):
    """Hourglass Transformer.

    A U-shaped transformer [CK]_ with neighborhood self-attention [NAT]_.

    References
    ----------
    .. [CK] Crowson, Katherine, et al. "Scalable high-resolution pixel-space image synthesis with
        hourglass diffusion transformers." ICML 2024, https://arxiv.org/abs/2401.11605
    .. [NAT] Hassani, A. et al. "Neighborhood Attention Transformer" CVPR, 2023, https://arxiv.org/abs/2204.07143

    """

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
        """Initialize the Hourglass Transformer.

        Parameters
        ----------
        n_dim
            Number of (spatial)dimensions of the input data.
        n_channels_in
            Number of channels in the input data.
        n_channels_out
            Number of channels in the output data.
        n_features
            Number of features in each stage.
        depths
            Number of layers in each stage.
        attention_neighborhood
            Neighborhood size for the neighborhood self-attention. If None, use global attention
            for that stage.
        n_heads
            Number of heads in each stage.
        cond_dim
            Number of dimensions of the conditioning tensor.
        """
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

        move_channels_last = RearrangeOp('batch  channels ...  -> batch ... channels')
        first_block = Sequential(
            move_channels_last,
            PixelUnshuffleDownsample(n_dim, n_channels_in, n_features_[0], downscale_factor=2, features_last=True),
        )
        dim_group = (tuple(range(-n_dim - 1, -1)),)
        encoder_blocks: list[Module] = []
        decoder_blocks: list[Module] = []
        merge_blocks: list[Module] = []
        down_blocks: list[Module] = []
        up_blocks: list[Module] = []
        for channels, depth, neighborhood, head in zip(
            n_features_[:-1],
            depths_[:-1],
            attention_neighborhood_[:-1],
            n_heads_[:-1],
            strict=True,
        ):
            encoder_blocks.append(
                SpatialTransformerBlock(
                    dim_groups=dim_group,
                    channels=channels,
                    depth=depth,
                    attention_neighborhood=neighborhood,
                    n_heads=head,
                    rope_embed_fraction=1.0,
                    cond_dim=cond_dim,
                    features_last=True,
                    norm='rms',
                )
            )
            decoder_blocks.append(
                SpatialTransformerBlock(
                    dim_groups=dim_group,
                    channels=channels,
                    depth=depth,
                    attention_neighborhood=neighborhood,
                    n_heads=head,
                    rope_embed_fraction=1.0,
                    cond_dim=cond_dim,
                    features_last=True,
                    norm='rms',
                )
            )
            merge_blocks.append(Interpolate())
        for channels, channels_next in pairwise(n_features_):
            down_blocks.append(
                PixelUnshuffleDownsample(n_dim, channels, channels_next, downscale_factor=2, features_last=True)
            )
            up_blocks.append(PixelShuffleUpsample(n_dim, channels_next, channels, upscale_factor=2, features_last=True))

        last_block = Sequential(
            PixelShuffleUpsample(n_dim, n_features_[-1], n_channels_out, upscale_factor=2, features_last=True),
            move_channels_last.H,  # moves channels back to front
        )
        middle_block = SpatialTransformerBlock(
            dim_groups=dim_group,
            channels=n_features_[-1],
            depth=depths_[-1],
            attention_neighborhood=attention_neighborhood_[-1],
            n_heads=n_heads_[-1],
            rope_embed_fraction=1.0,
            cond_dim=cond_dim,
            features_last=True,
            norm='rms',
        )
        encoder = UNetEncoder(first_block, encoder_blocks, down_blocks, middle_block)
        decoder = UNetDecoder(decoder_blocks, up_blocks, merge_blocks, last_block)

        super().__init__(encoder, decoder)
