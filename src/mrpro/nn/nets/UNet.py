"""UNet variants."""

from collections.abc import Sequence
from functools import partial

import torch
from sympy import Identity
from torch.nn import Module, ModuleList

from mrpro.nn.CondMixin import call_with_cond


class UNetEncoder(Module):
    """Encoder."""

    def __init__(
        self,
        first_block: Module,
        encoder_blocks: Sequence[Module],
        down_blocks: Sequence[Module],
        middle_block: Module,
    ) -> None:
        """Initialize the UNetEncoder."""
        super().__init__()
        self.first = first_block
        """The first block. Should expand from the number of input channels."""

        self.encoder_blocks = ModuleList(encoder_blocks)
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
        for block, down in zip(self.encoder_blocks, self.down_blocks, strict=True):
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
        return super().__call__(x, cond)


class UNetDecoder(Module):
    """Decoder."""

    def __init__(
        self,
        decoder_blocks: Sequence[Module],
        up_blocks: Sequence[Module],
        concat_blocks: Sequence[Module],
        last_block: Module,
    ) -> None:
        """Initialize the UNetDecoder."""
        super().__init__()
        self.decoder_blocks = ModuleList(decoder_blocks)
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
        for block, up, concat, h in zip(
            self.decoder_blocks, self.up_blocks, self.concat_blocks, hs[-2::-1], strict=True
        ):
            x = call(up, x)
            x = concat(x, h)
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


class UNet(UNetBase):
    """UNet.

    U-shaped convolutional network [UNET]_ with optional patch attention.
    Inspired by the OpenAi DDPM UNet/Latent Diffusion UNet [LDM]_.

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
        n_features: Sequence[int],
        n_heads: Sequence[int],
        n_blocks: int | Sequence[int],
        cond_dim: int,
        num_blocks: int,
        padding_modes: str | Sequence[str],
    ) -> None:
        """Initialize the UNet."""
        super().__init__()


class AttentionUNet(UNet):
    """UNet with attention gates.

    References
    ----------
    .. [OKT18] Oktay, Ozan, et al. "Attention U-net: Learning where to look for the pancreas." MIDL (2018).
      https://arxiv.org/abs/1804.03999
    """


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
