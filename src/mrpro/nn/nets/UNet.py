"""UNet variants."""

from collections.abc import Sequence
from functools import partial
from itertools import pairwise

import torch
from torch.nn import Identity, Module, ModuleList, ReLU, SiLU

from mrpro.nn.AttentionGate import AttentionGate
from mrpro.nn.CondMixin import call_with_cond
from mrpro.nn.FiLM import FiLM
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.join import Concat
from mrpro.nn.ndmodules import ConvND, MaxPoolND
from mrpro.nn.PermutedBlock import PermutedBlock
from mrpro.nn.ResBlock import ResBlock
from mrpro.nn.SeparableResBlock import SeparableResBlock  # Assuming SeparableResBlock is here
from mrpro.nn.Sequential import Sequential
from mrpro.nn.SpatialTransformerBlock import SpatialTransformerBlock
from mrpro.nn.Upsample import Upsample


class UNetEncoder(Module):
    """Encoder."""

    def __init__(
        self,
        first_block: Module,
        blocks: Sequence[Module],
        down_blocks: Sequence[Module],
        middle_block: Module,
    ) -> None:
        """Initialize the UNetEncoder."""
        super().__init__()
        self.first = first_block
        """The first block. Should expand from the number of input channels."""

        self.blocks = ModuleList(blocks)
        """The encoder blocks. Order is highest resolution to lowest resolution."""

        self.down_blocks = ModuleList(down_blocks)
        """The downsampling blocks"""

        self.middle_block = middle_block
        """Also called bottleneck block"""

    def __len__(self):
        """Get the number of resolutions levels."""
        return len(self.down_blocks) + 1

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """Apply to Network."""
        call = partial(call_with_cond, cond=cond)

        x = call(self.first, x)

        xs = []
        for block, down in zip(self.blocks, self.down_blocks, strict=True):
            x = call(block, x)
            xs.append(x)
            x = call(down, x)

        x = call(self.middle_block, x)

        return (*xs, x)

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """Apply to Network.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The conditioning tensor.

        Returns
        -------
            The tensors at the different resolutions, highest resolution first.
        """
        return super().__call__(x, cond=cond)


class UNetDecoder(Module):
    """Decoder."""

    def __init__(
        self,
        blocks: Sequence[Module],
        up_blocks: Sequence[Module],
        concat_blocks: Sequence[Module],
        last_block: Module,
    ) -> None:
        """Initialize the UNetDecoder."""
        super().__init__()
        self.blocks = ModuleList(blocks)
        """The decoder blocks. Order is lowest resolution to highest resolution."""

        self.up_blocks = ModuleList(up_blocks)
        """The upsampling blocks"""

        self.concat_blocks = ModuleList(concat_blocks)
        """Joins the skip connections with the upsampled features from a lower resolution level"""

        self.last_block = last_block
        """The last block. Should reduce to the number of output channels."""

    def __len__(self):
        """Get the number of resolutions levels."""
        return len(self.up_blocks) + 1

    def forward(self, hs: tuple[torch.Tensor, ...], *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply to Network."""
        call = partial(call_with_cond, cond=cond)

        x = hs[-1]  # lowest resolution, from middle block
        for block, up, concat, h in zip(self.blocks, self.up_blocks, self.concat_blocks, hs[-2::-1], strict=True):
            x = call(up, x)
            x = concat(h, x)
            x = call(block, x)
        x = call(self.last_block, x)
        return x

    def __call__(self, hs: tuple[torch.Tensor, ...], *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply to Network.

        Parameters
        ----------
        hs
            The tensors at the different resolutions, highest resolution first.
        cond
            The conditioning tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(hs, cond=cond)


class UNetBase(Module):
    """Base class for U-shaped networks."""

    def __init__(self, encoder: UNetEncoder, decoder: UNetDecoder, skip_blocks: Sequence[Module] | None = None) -> None:
        """Initialize the UNetBase."""
        super().__init__()
        self.encoder = encoder
        """The encoder."""

        self.decoder = decoder
        """The decoder."""

        self.skip_blocks = ModuleList()
        """Modifications of the skip connections."""

        # if len(decoder) != len(encoder):
        #    raise ValueError(
        #        'The number of resolutions in the encoder and decoder must be the same, '
        #        f'got {len(decoder)} and {len(encoder)}'
        #    )

        if skip_blocks is None:
            self.skip_blocks.extend(Identity() for _ in range(len(decoder)))
        elif len(skip_blocks) != len(decoder):
            raise ValueError(
                f'The number of skip blocks must be the same as the number of resolutions, '
                f'got {len(skip_blocks)} and {len(encoder)}'
            )
        else:
            self.skip_blocks.extend(skip_blocks)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply to Network."""
        xs = self.encoder(x, cond=cond)
        xs = tuple(
            call_with_cond(self.skip_blocks[i], x, cond=cond) if i < len(self.skip_blocks) else x
            for i, x in enumerate(xs)
        )
        x = self.decoder(xs, cond=cond)
        return x

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply to Network.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The conditioning tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, cond=cond)


class BasicUNet(UNetBase):
    """Basic UNet.

    A Basic UNet with residual blocks, convolutional downsampling, and nearest neighbor upsampling.

    References
    ----------
    .. [UNET] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image
       segmentation MICCAI 2015. https://arxiv.org/abs/1505.04597
    """

    def __init__(self, dim: int, channels_in: int, channels_out: int, n_features: Sequence[int], cond_dim: int):
        """Initialize the BasicUNet."""
        encoder_blocks: list[Module] = []
        decoder_blocks: list[Module] = []
        down_blocks: list[Module] = []
        up_blocks: list[Module] = []
        concat_blocks: list[Module] = []
        for n_feat, n_feat_next in pairwise(n_features):
            encoder_blocks.append(ResBlock(dim, n_feat, n_feat, cond_dim))
            decoder_blocks.append(ResBlock(dim, 2 * n_feat, n_feat, cond_dim))
            down_blocks.append(ConvND(dim)(n_feat, n_feat_next, 3, stride=2, padding=1))
            up_blocks.append(
                Sequential(
                    Upsample(tuple(range(-dim, 0)), scale_factor=2), ConvND(dim)(n_feat_next, n_feat, 3, padding=1)
                )
            )
            concat_blocks.append(Concat())
        up_blocks = up_blocks[::-1]
        decoder_blocks = decoder_blocks[::-1]
        first_block = ConvND(dim)(channels_in, n_features[0], 3, padding=1)
        last_block = Sequential(
            GroupNorm(n_features[0]), SiLU(), ConvND(dim)(n_features[0], channels_out, 3, padding=1)
        )
        middle_block = ResBlock(dim, n_features[-1], n_features[-1], cond_dim)
        encoder = UNetEncoder(first_block, encoder_blocks, down_blocks, middle_block)
        decoder = UNetDecoder(decoder_blocks, up_blocks, concat_blocks, last_block)
        super().__init__(encoder, decoder)


class UNet(UNetBase):
    """UNet.

    U-shaped convolutional network with optional patch attention.
    Inspired by the OpenAi DDPM UNet/Latent Diffusion UNet [LDM]_,
    significant differences to the vanilla UNet [UNET]_ include:
       - Spatial transformer blocks
       - Convolutional downsampling, nearest neighbor upsampling
       - Residual convolution blocks with pre-act group normalization and SiLU activation

    References
    ----------
    .. [UNET] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image
       segmentation MICCAI 2015. https://arxiv.org/abs/1505.04597
    .. [LDM] https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
    """

    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        attention_depths: Sequence[int] = (-1,),
        n_features: Sequence[int] = (64, 128, 192, 256),
        n_heads: int = 8,
        cond_dim: int = 0,
        encoder_blocks_per_scale: int = 2,
    ) -> None:
        """Initialize the UNet.

        Parameters
        ----------
        dim
            Spatial dimension of the input tensor.
        channels_in
            Number of channels in the input tensor.
        channels_out
            Number of channels in the output tensor.
        attention_depths
            The depths at which to apply attention.
        n_features
            Number of features at each resolution level. The length determines the number of resolution levels.
        n_heads
            Number of attention heads.
        cond_dim
            Number of channels in the conditioning tensor. If 0, no conditioning is applied.
        encoder_blocks_per_scale
            Number of encoder blocks per resolution level. The number of decoder blocks is one more.
        """
        depth = len(n_features)
        if not all(-depth <= d < depth for d in attention_depths):
            raise ValueError(
                f'attention_depths must be in the range [-depth, depth], got {attention_depths=} for {depth=}'
            )
        attention_depths = tuple(d % depth for d in attention_depths)
        if len(attention_depths) != len(set(attention_depths)):
            raise ValueError(f'attention_depths must be unique, got {attention_depths=}')

        def attention_block(channels: int) -> Module:
            dim_groups = (tuple(range(-dim, 0)),)
            return SpatialTransformerBlock(dim_groups, channels, n_heads, cond_dim=cond_dim)

        def blocks(channels_in: int, channels_out: int, attention: bool) -> Module:
            blocks = Sequential()
            for _ in range(encoder_blocks_per_scale):
                blocks.append(ResBlock(dim, channels_in, channels_out, cond_dim))
                if attention:
                    blocks.append(attention_block(channels_out))
                channels_in = channels_out
            return blocks

        encoder_blocks: list[Module] = [ConvND(dim)(channels_in, n_features[0], 3, padding=1)]
        down_blocks: list[Module] = [Identity()]
        decoder_blocks: list[Module] = []
        up_blocks: list[Module] = []

        for i_level, (n_feat, n_feat_next) in enumerate(pairwise(n_features)):
            encoder_blocks.append(blocks(n_feat, n_feat, i_level in attention_depths))
            down_blocks.append(ConvND(dim)(n_feat, n_feat_next, 3, stride=2, padding=1))
            decoder_blocks.append(blocks(n_feat_next + n_feat, n_feat, i_level in attention_depths))
            up_blocks.append(Upsample(tuple(range(-dim, 0)), scale_factor=2))

        middle_block = Sequential(
            ResBlock(dim, n_feat_next, n_feat_next, cond_dim),
            ResBlock(dim, n_feat_next, n_feat_next, cond_dim),
        )
        if depth - 1 in attention_depths:
            middle_block.insert(1, attention_block(n_feat_next))
        first_block = ConvND(dim)(channels_in, n_features[0], 3, padding=1)
        encoder = UNetEncoder(first_block, encoder_blocks, down_blocks, middle_block)

        decoder_blocks, up_blocks = decoder_blocks[::-1], up_blocks[::-1]
        last_block = Sequential(
            SiLU(),
            ConvND(dim)(n_features[0], channels_out, 3, padding=1),
        )
        concat_blocks = [Concat() for _ in range(len(decoder_blocks))]
        decoder = UNetDecoder(decoder_blocks, up_blocks, concat_blocks, last_block)

        super().__init__(encoder, decoder)


class AttentionGatedUNet(UNetBase):
    """UNet with attention gates.

    Basic UNet with attention gating of the skip signals by the lower resolution features [OKT18]_.

    References
    ----------
    .. [OKT18] Oktay, Ozan, et al. "Attention U-net: Learning where to look for the pancreas." MIDL (2018).
      https://arxiv.org/abs/1804.03999
    """

    def __init__(self, dim: int, channels_in: int, channels_out: int, n_features: Sequence[int], cond_dim: int = 0):
        """Initialize the AttentionGatedUNet.

        Parameters
        ----------
        dim
            Spatial dimension of the input tensor.
        channels_in
            Number of channels in the input tensor.
        channels_out
            Number of channels in the output tensor.
        n_features
            Number of features at each resolution level. The length determines the number of resolution levels.
        cond_dim
            Number of channels in the conditioning tensor. If 0, no conditioning is applied.
        """

        def block(channels_in: int, channels_out: int) -> Module:
            block = Sequential(
                ConvND(dim)(channels_in, channels_out, 3, padding=1),
                ReLU(True),
                ConvND(dim)(channels_out, channels_out, 3, padding=1),
                ReLU(True),
            )
            if cond_dim > 0:
                block.insert(2, FiLM(channels_out, cond_dim))
            return block

        encoder_blocks: list[Module] = []
        down_blocks: list[Module] = []
        n_feat_old = channels_in
        for n_feat in n_features[:-1]:
            encoder_blocks.append(block(n_feat_old, n_feat))
            down_blocks.append(MaxPoolND(dim)(2))
            n_feat_old = n_feat
        middle_block = block(n_features[-2], n_features[-1])
        encoder = UNetEncoder(Identity(), encoder_blocks, down_blocks, middle_block)

        concat_blocks = []
        decoder_blocks: list[Module] = []
        up_blocks: list[Module] = []
        for n_feat, n_feat_skip in pairwise(n_features[::-1]):
            concat_blocks.append(AttentionGate(dim, n_feat, n_feat_skip, n_feat_skip, concatenate=True))
            decoder_blocks.append(block(n_feat + n_feat_skip, n_feat_skip))
            up_blocks.append(Upsample(range(-dim, 0), scale_factor=2))
        last_block = ConvND(dim)(n_features[0], channels_out, 1)
        decoder = UNetDecoder(decoder_blocks, up_blocks, concat_blocks, last_block)

        super().__init__(encoder, decoder)


class SeparableUNet(UNetBase):
    """UNet with separable convolutions and attention, and grouped downsampling."""

    def __init__(
        self,
        dim: int,
        dim_groups: Sequence[tuple[int, ...]],
        channels_in: int,
        channels_out: int,
        n_features: Sequence[int] = (64, 128, 256, 512),
        cond_dim: int = 0,
        encoder_blocks_per_scale: int = 2,
        attention_depths: Sequence[int] = (-1,),
        n_heads: int = 8,
        downsample_dims: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """
        Initialize the SeparableUNet.

        Parameters
        ----------
        dim
            Total number of non batch, non channel dimensions.
            E.g., 2 for 2D images, 3 for 3D volumes or 2D+time for 2D+time images.
        dim_groups
            A list of tuples, where each tuple contains the spatial dimension
            indices for one separable convolution. Each group must contain fewer than 3 dimensions.
        channels_in
            Number of channels in the input tensor.
        channels_out
            Number of channels in the output tensor.
        n_features
            Number of features at each resolution level.
        cond_dim
            Number of channels in the conditioning tensor.
        encoder_blocks_per_scale
            Number of encoder blocks per resolution level.
        attention_depths
            The depths at which to apply attention.
        n_heads
            Number of attention heads.
        downsample_dims
            Sequence specifying which absolute spatial dimensions to downsample
            at each encoder level. If None, all dimensions in `dim_groups` are combined
            and downsampled at each level.
            If a downsampling step contains more than 3 dimensions, downsampling is performed separately for each
            dimension. If the length of the sequence is less than the number of resolution levels, the sequence is
            repeated. E.g., ``((-1,-2), (-1,-2,-3))`` for 3D data: first level downsamples x,y; second level x,y,z;
            third level x,y.


        """
        depth = len(n_features)
        for group in dim_groups:
            if len(group) > 3:
                raise ValueError(f'dim_group {group} can at most contain 3 dimensions. Split it into multiple groups.')
            if any(d > dim + 2 or d < -dim for d in group):
                raise ValueError(f'dim_group {group} contains dimensions that are out of range for dim={dim}')

        attention_depths = tuple(d % depth for d in attention_depths)
        if downsample_dims is None:
            all_spatial_dims = tuple(sorted(set(d if d < 0 else d - dim - 2 for group in dim_groups for d in group)))
            downsample_dims = (all_spatial_dims,) * (depth - 1)

        def downsampler(level_dims, c_in, c_out) -> Module:
            if len(level_dims) > 3:
                sequence = Sequential(*(downsampler(d[0], c_in, c_out) for d in level_dims))
                for d in level_dims[1:]:
                    sequence.append(downsampler(d, c_out, c_out))
                return sequence
            return PermutedBlock(level_dims, ConvND(len(level_dims))(c_in, c_out, 3, stride=2, padding=1))

        def upsampler(level_dims, c_in, c_out) -> Module:
            return Upsample(level_dims, scale_factor=2)

        def block(c_in: int, c_out: int, apply_attention: bool) -> Module:
            res_block = SeparableResBlock(dim_groups, c_in, c_out, cond_dim)
            if not apply_attention:
                return res_block
            attn_block = SpatialTransformerBlock(dim_groups, c_out, n_heads, cond_dim=cond_dim)
            return Sequential(res_block, attn_block)

        # --- Module Construction ---
        first_block = PermutedBlock(
            all_spatial_dims, ConvND(len(all_spatial_dims))(channels_in, n_features[0], 3, padding=1)
        )

        # -- Encoder --
        encoder_blocks, down_blocks, skip_features = [], [], []
        c_feat = n_features[0]
        for i_level, n_feat_level in enumerate(n_features):
            for _ in range(encoder_blocks_per_scale):
                encoder_blocks.append(block(c_feat, n_feat_level, i_level in attention_depths))
                c_feat = n_feat_level
                skip_features.append(c_feat)
            if i_level < depth - 1:
                down_blocks.append(downsampler(downsample_dims_per_level[i_level], c_feat, n_features[i_level + 1]))
                c_feat = n_features[i_level + 1]

        # -- Middle & Encoder Finalization --
        middle_block = Sequential(
            block(c_feat, c_feat, depth - 1 in attention_depths),
            block(c_feat, c_feat, depth - 1 in attention_depths),
        )
        encoder = UNetEncoder(first_block, encoder_blocks, down_blocks, middle_block)

        # -- Decoder --
        decoder_blocks, up_blocks = [], []
        for i_level in reversed(range(depth)):
            n_feat_level = n_features[i_level]
            if i_level > 0:
                up_blocks.append(upsampler(downsample_dims_per_level[i_level - 1], c_feat, n_feat_level))
            for _ in range(encoder_blocks_per_scale + 1):
                skip_c = skip_features.pop()
                decoder_blocks.append(block(c_feat + skip_c, n_feat_level, i_level in attention_depths))
                c_feat = n_feat_level

        decoder_blocks.reverse()
        up_blocks.reverse()

        # -- Decoder Finalization --
        concat_blocks = [Concat()] * len(decoder_blocks)
        last_block = Sequential(
            GroupNorm(n_features[0]),
            SiLU(),
            PermutedBlock(
                all_spatial_dims,
                ConvND(len(all_spatial_dims))(n_features[0], channels_out, 3, padding=1),
            ),
        )
        decoder = UNetDecoder(decoder_blocks, up_blocks, concat_blocks, last_block)

        super().__init__(encoder, decoder)
