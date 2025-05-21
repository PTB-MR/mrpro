"""UNet variants."""

from collections.abc import Sequence
from functools import partial

import torch
from torch.nn import Identity, Module, ModuleList

from mrpro.nn.CondMixin import call_with_cond


class UNetBase(Module):
    """Base class for U-shaped networks."""

    def __init__(self) -> None:
        """Initialize the UNetBase."""
        super().__init__()
        self.input_blocks = ModuleList()
        """The encoder blocks. Order is highest resolution to lowest resolution."""

        self.down_blocks = ModuleList()
        """The downsampling blocks"""

        self.skip_blocks = ModuleList()
        """Modifications to the skip connections"""

        self.middle_block: Module = Identity()
        """Also called bottleneck block"""

        self.output_blocks = ModuleList()
        """Also called decoder blocks. Order is lowest resolution to highest resolution."""

        self.up_blocks = ModuleList()
        """The upsampling blocks"""

        self.concat_blocks = ModuleList()
        """Joins the skip connections with the upsampled features from a lower resolution level"""

        self.last: Module = Identity()
        """The last block. Should reduce to the number of output channels."""

        self.first: Module = Identity()
        """The first block. Should expand from the number of input channels."""

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply to Network."""
        call = partial(call_with_cond, cond=cond)
        x = call(self.first, x)
        xs = []
        for block, down, skip in zip(self.input_blocks, self.down_blocks, self.skip_blocks, strict=True):
            x = call(block, x)
            xs.append(call(skip, x))
            x = call(down, x)
        x = call(self.middle_block, x)
        for block, up, concat in zip(self.output_blocks, self.up_blocks, self.concat_blocks, strict=True):
            x = call(up, x)
            x = concat(x, xs.pop())
            x = call(block, x)
        return call(self.last, x)

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
        return super().__call__(x, cond)


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
