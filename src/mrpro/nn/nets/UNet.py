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
from mrpro.nn.ResBlock import ResBlock
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
            up_blocks.append(Sequential(Upsample(dim, scale_factor=2), ConvND(dim)(n_feat_next, n_feat, 3, padding=1)))
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
       - Spatial attention
       - Multiple skip connections per resolution
       - Convolutional downsampling, nearest neighbor upsampling
       - Residual convolution blocks
       - Group normalization
       - SiLU activation

    References
    ----------
    .. [UNET] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image
       segmentation MICCAI 2015. https://arxiv.org/abs/1505.04597
    .. [LDM] https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        attention_depths: Sequence[int] = (-1, -2),
        n_features: Sequence[int] = (64, 128, 192, 256),
        n_heads: int = 4,
        cond_dim: int = 0,
        encoder_blocks_per_scale: int = 2,
    ) -> None:
        """Initialize the UNet."""
        depth = len(n_features)
        if not all(-depth <= d < depth for d in attention_depths):
            raise ValueError(
                f'attention_depths must be in the range [-depth, depth], got {attention_depths=} for {depth=}'
            )
        attention_depths = tuple(d % depth for d in attention_depths)
        if len(attention_depths) != len(set(attention_depths)):
            raise ValueError(f'attention_depths must be unique, got {attention_depths=}')

        def attention_block(channels: int) -> Module:
            return SpatialTransformerBlock(
                dim, channels, n_heads, channels_per_head=channels // n_heads, cond_dim=cond_dim
            )

        def block(channels_in: int, channels_out: int, attention: bool) -> Module:
            if not attention:
                return ResBlock(dim, channels_in, channels_out, cond_dim)
            return Sequential(ResBlock(dim, channels_in, channels_out, cond_dim), attention_block(channels_out))

        first_block = ConvND(dim)(in_channels, n_features[0], 3, padding=1)
        encoder_blocks: list[Module] = []
        down_blocks: list[Module] = []
        skip_features = []
        n_feat_old = n_features[0]
        for i_level, n_feat in enumerate(n_features):
            encoder_blocks.append(Identity())
            skip_features.append(n_feat_old)
            for _ in range(encoder_blocks_per_scale):
                encoder_blocks.append(block(n_feat_old, n_feat, attention=i_level in attention_depths))
                n_feat_old = n_feat
                down_blocks.append(Identity())
                skip_features.append(n_feat_old)
            down_blocks.append(ConvND(dim)(n_feat, n_feat, 3, stride=2, padding=1))
        down_blocks[-1] = Identity()
        middle_block = Sequential(
            ResBlock(dim, n_features[-1], n_features[-1], cond_dim),
            ResBlock(dim, n_features[-1], n_features[-1], cond_dim),
        )
        if i_level in attention_depths:
            middle_block.insert(1, attention_block(n_features[-1]))
        encoder = UNetEncoder(first_block, encoder_blocks, down_blocks, middle_block)

        decoder_blocks: list[Module] = []
        up_blocks: list[Module] = [Identity()]
        for i_level, n_feat in reversed(list(enumerate(n_features))):
            decoder_blocks.append(
                block(n_feat_old + skip_features.pop(), n_feat, attention=i_level in attention_depths)
            )
            n_feat_old = n_feat
            for _ in range(encoder_blocks_per_scale):
                decoder_blocks.append(
                    block(n_feat_old + skip_features.pop(), n_feat, attention=i_level in attention_depths)
                )
                n_feat_old = n_feat

                up_blocks.append(Identity())
                n_feat_old = n_feat
            up_blocks.append(Upsample(dim, scale_factor=2))
        up_blocks.pop()
        concat_blocks = [Concat()] * len(decoder_blocks)
        last_block = Sequential(
            GroupNorm(n_features[0]), SiLU(), ConvND(dim)(n_features[0], out_channels, 3, padding=1)
        )
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
            up_blocks.append(Upsample(dim, scale_factor=2))
        last_block = ConvND(dim)(n_features[0], channels_out, 1)
        decoder = UNetDecoder(decoder_blocks, up_blocks, concat_blocks, last_block)

        super().__init__(encoder, decoder)


class SeparableUNet(UNetBase):
    """UNet where blocks apply separable convolutions in different dimensions.

    Based on the pseudo-3D residual network of [QUI]_, [TRAN]_ and the residual blocks of [ZIM]_.

    References
    ----------
    .. [TRAN] Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. A closer look at spatiotemporal
       convolutions for action recognition. CVPR 2018. https://arxiv.org/abs/1711.11248
    .. [QUI] Qiu, Z., Yao, T., & Mei, T. Learning spatio-temporal representation with pseudo-3d residual networks.
       ICCV 2017. https://arxiv.org/abs/1711.10305
    .. [ZIM] Zimmermann, F. F., & Kofler, A. (2023, October). NoSENSE: Learned unrolled cardiac MRI reconstruction
       without explicit sensitivity maps. STACOM MICCAI 2023. https://arxiv.org/abs/2309.15608
    """
