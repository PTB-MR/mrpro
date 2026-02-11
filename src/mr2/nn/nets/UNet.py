"""UNet variants."""

from collections.abc import Sequence
from functools import partial
from itertools import pairwise

import torch
from torch.nn import Identity, Module, ModuleList, ReLU, SiLU

from mr2.nn.attention.AttentionGate import AttentionGate
from mr2.nn.attention.SpatialTransformerBlock import SpatialTransformerBlock
from mr2.nn.CondMixin import call_with_cond
from mr2.nn.FiLM import FiLM
from mr2.nn.join import Concat
from mr2.nn.ndmodules import convND, maxPoolND
from mr2.nn.ResBlock import ResBlock
from mr2.nn.Sequential import Sequential
from mr2.nn.Upsample import Upsample


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

        if len(decoder) != len(encoder):
            raise ValueError(
                'The number of resolutions in the encoder and decoder must be the same, '
                f'got {len(decoder)} and {len(encoder)}'
            )

        if skip_blocks is None:
            self.skip_blocks.extend(Identity() for _ in range(len(decoder)))
        elif len(skip_blocks) != len(decoder):
            raise ValueError(
                f'The number of skip blocks must be the same as the number of resolutions, '
                f'got {len(skip_blocks)} and {len(encoder)}'
            )
        else:
            self.skip_blocks.extend(skip_blocks)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply to Network."""
        xs = self.encoder(x, cond=cond)
        xs = tuple(
            call_with_cond(self.skip_blocks[i], x, cond=cond) if i < len(self.skip_blocks) else x
            for i, x in enumerate(xs)
        )
        x = self.decoder(xs, cond=cond)
        return x

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
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


class UNet(UNetBase):
    """UNet.

    U-shaped convolutional network with optional patch attention.
    Inspired by [NOSENSE_] and the OpenAi DDPM UNet/Latent Diffusion UNet [LDM]_.
    significant differences to the vanilla UNet [UNET]_ include:
       - Spatial transformer blocks
       - Convolutional downsampling, nearest neighbor upsampling
       - Residual convolution blocks with pre-act group normalization and SiLU activation


    References
    ----------
    .. [UNET] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image
       segmentation MICCAI 2015. https://arxiv.org/abs/1505.04597
    .. [LDM] https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
    .. [NOSENSE] Zimmermann, FF, and Kofler, Andreas. "NoSENSE: Learned unrolled cardiac MRI reconstruction without
        explicit sensitivity maps." STACOM 2023. https://github.com/fzimmermann89/CMRxRecon/blob/master/src/cmrxrecon/nets/unet.py

    """

    def __init__(
        self,
        n_dim: int,
        n_channels_in: int,
        n_channels_out: int,
        attention_depths: Sequence[int] = (-1,),
        n_features: Sequence[int] = (64, 128, 192, 256),
        n_heads: int = 8,
        cond_dim: int = 0,
        encoder_blocks_per_scale: int = 2,
    ) -> None:
        """Initialize the UNet.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels_in
            The number of channels in the input tensor.
        n_channels_out
            The number of channels in the output tensor.
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
            dim_groups = (tuple(range(-n_dim, 0)),)
            return SpatialTransformerBlock(dim_groups, channels, n_heads, cond_dim=cond_dim)

        def blocks(channels_in: int, channels_out: int, attention: bool) -> Module:
            blocks = Sequential()
            for _ in range(encoder_blocks_per_scale):
                blocks.append(ResBlock(n_dim, channels_in, channels_out, cond_dim))
                if attention:
                    blocks.append(attention_block(channels_out))
                channels_in = channels_out
            return blocks

        encoder_blocks: list[Module] = []
        down_blocks: list[Module] = []
        decoder_blocks: list[Module] = []
        up_blocks: list[Module] = []

        for i_level, (n_feat, n_feat_next) in enumerate(pairwise(n_features)):
            encoder_blocks.append(blocks(n_feat, n_feat, i_level in attention_depths))
            down_blocks.append(convND(n_dim)(n_feat, n_feat_next, 3, stride=2, padding=1))
            decoder_blocks.append(blocks(n_feat_next + n_feat, n_feat, i_level in attention_depths))
            up_blocks.append(Upsample(tuple(range(-n_dim, 0)), scale_factor=2))

        middle_block = Sequential(
            ResBlock(n_dim, n_feat_next, n_feat_next, cond_dim),
            ResBlock(n_dim, n_feat_next, n_feat_next, cond_dim),
        )
        if depth - 1 in attention_depths:
            middle_block.insert(1, attention_block(n_feat_next))
        first_block = convND(n_dim)(n_channels_in, n_features[0], 3, padding=1)
        encoder = UNetEncoder(first_block, encoder_blocks, down_blocks, middle_block)

        decoder_blocks, up_blocks = decoder_blocks[::-1], up_blocks[::-1]
        last_block = Sequential(
            SiLU(),
            convND(n_dim)(n_features[0], n_channels_out, 3, padding=1),
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

    def __init__(
        self, n_dim: int, n_channels_in: int, n_channels_out: int, n_features: Sequence[int], cond_dim: int = 0
    ):
        """Initialize the AttentionGatedUNet.

        Parameters
        ----------
        n_dim
            The number of spatial dimensions of the input tensor.
        n_channels_in
            The number of channels in the input tensor.
        n_channels_out
            The number of channels in the output tensor.
        n_features
            Number of features at each resolution level. The length determines the number of resolution levels.
        cond_dim
            Number of channels in the conditioning tensor. If 0, no conditioning is applied.
        """

        def block(channels_in: int, channels_out: int) -> Module:
            block = Sequential(
                convND(n_dim)(channels_in, channels_out, 3, padding=1),
                ReLU(True),
                convND(n_dim)(channels_out, channels_out, 3, padding=1),
                ReLU(True),
            )
            if cond_dim > 0:
                block.insert(2, FiLM(channels_out, cond_dim))
            return block

        encoder_blocks: list[Module] = []
        down_blocks: list[Module] = []
        n_feat_old = n_channels_in
        for n_feat in n_features[:-1]:
            encoder_blocks.append(block(n_feat_old, n_feat))
            down_blocks.append(maxPoolND(n_dim)(2))
            n_feat_old = n_feat
        middle_block = block(n_features[-2], n_features[-1])
        encoder = UNetEncoder(Identity(), encoder_blocks, down_blocks, middle_block)

        concat_blocks = []
        decoder_blocks: list[Module] = []
        up_blocks: list[Module] = []
        for n_feat, n_feat_skip in pairwise(n_features[::-1]):
            concat_blocks.append(AttentionGate(n_dim, n_feat, n_feat_skip, n_feat_skip, concatenate=True))
            decoder_blocks.append(block(n_feat + n_feat_skip, n_feat_skip))
            up_blocks.append(Upsample(range(-n_dim, 0), scale_factor=2))
        last_block = convND(n_dim)(n_features[0], n_channels_out, 1)
        decoder = UNetDecoder(decoder_blocks, up_blocks, concat_blocks, last_block)

        super().__init__(encoder, decoder)
