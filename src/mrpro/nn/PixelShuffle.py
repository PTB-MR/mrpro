"""ND-version of PixelShuffle and PixelUnshuffle."""

import torch
from torch.nn import Module

from mrpro.nn.ndmodules import ConvND


class PixelUnshuffle(Module):
    """ND-version of PixelUnshuffle downscaling."""

    def __init__(self, downscale_factor: int):
        """Initialize PixelUnshuffle.

        Reduces spatial dimensions and increases the channel number by reshaping.
        The first dimension is considered a batch dimension, the second dimension
        the channel dimension, and the remaining dimensions the spatial dimensions that are downscaled.

        See `mrpro.nn.PixelShuffle` for the inverse operation.

        Parameters
        ----------
        downscale_factor : int
            The factor by which to downscale the input tensor.
        """
        super().__init__()
        self.downscale_factor = downscale_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Downscale the input.

        Parameters
        ----------
        x
            Tensor of shape `batch, channels, *spatial_dims`

        Returns
        -------
        Tensor of shape `batch, channels * downscale_factor**dim, *spatial_dims/downscale_factor`
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downscale the input."""
        dim = x.ndim - 2
        if dim == 2:  # fast path for 2D
            return torch.nn.functional.pixel_unshuffle(x, self.downscale_factor)

        new_shape = list(x.shape[:2])
        source_positions = []
        for i, old in enumerate(x.shape[2:]):
            new_shape.append(old // self.downscale_factor)
            new_shape.append(self.downscale_factor)
            source_positions.append(2 + 2 * i)

        x = x.view(new_shape)
        x = x.moveaxis(source_positions, tuple(range(-dim, 0)))
        x = x.flatten(1, -dim - 1)
        return x


class PixelUnshuffleDownsample(Module):
    """PixelUnshuffle Downsampling.

    PixelUnshuffle followed by a convolution. Optionally uses a residual connection [DCAE]_

    References
    ----------
    .. [DCAE] Chen et al. Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models. ICLR 2025
       https://arxiv.org/abs/2410.10733
    """

    def __init__(
        self, dim: int, channels_in: int, channels_out: int, downscale_factor: int = 2, residual: bool = False
    ):
        """Initialize a PixelUnshuffleDownsample layer.

        Parameters
        ----------
        dim : int
            Dimension of the input tensor.
        channels_in : int
            Number of channels in the input tensor.
        channels_out : int
            Number of channels in the output tensor.
        downscale_factor : int, optional
            Factor by which to downscale the input tensor.
        residual : bool, optional
            Whether to use a residual connection as proposed in [DCAE]_.
        """
        super().__init__()
        self.pixel_unshuffle = PixelUnshuffle(downscale_factor)
        out_ratio = downscale_factor**dim
        if channels_out % out_ratio != 0:
            raise ValueError(f'channels_out must be divisible by downscale_factor**{dim}.')
        self.conv = ConvND(dim)(channels_in, channels_out // out_ratio, kernel_size=3, padding='same')
        self.residual = residual

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling.

        Parameters
        ----------
        x
            Tensor of shape `batch, channels_in, *spatial_dims`

        Returns
        -------
            Tensor of shape `batch, channels_out, *spatial_dims/downscale_factor`
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling."""
        h = self.conv(x)
        h = self.pixel_unshuffle(h)

        if self.residual:
            x = self.pixel_unshuffle(x)
            h = h + x.unflatten(1, (h.shape[1], -1)).mean(2)
        return h


class PixelShuffleUpsample(Module):
    """PixelShuffle Upsampling.

    Convolution followed by PixelShuffle. Optionally uses a residual connection [DCAE]_

    References
    ----------
    .. [DCAE] Chen et al. Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models. ICLR 2025
       https://arxiv.org/abs/2410.10733
    """

    def __init__(self, dim: int, channels_in: int, channels_out: int, upscale_factor: int = 2, residual: bool = False):
        """Initialize a PixelShuffleUpsample layer.

        Parameters
        ----------
        dim : int
            Dimension of the input tensor.
        channels_in : int
            Number of channels in the input tensor.
        channels_out : int
            Number of channels in the output tensor.
        upscale_factor : int, optional
            Factor by which to upscale the input tensor.
        residual : bool, optional
            Whether to use a residual connection as proposed in [DCAE]_.
        """
        super().__init__()
        self.conv = ConvND(dim)(channels_in, channels_out * upscale_factor**dim, kernel_size=3, padding='same')
        self.pixel_shuffle = PixelShuffle(upscale_factor)
        self.residual = residual

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling.

        Parameters
        ----------
        x
            Tensor of shape `batch, channels_in, *spatial_dims`

        Returns
        -------
        Tensor of shape `batch, channels_out, *spatial_dims * upscale_factor`
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling."""
        h = self.conv(x)
        if self.residual:
            h = h + x.repeat_interleave(h.shape[1] // x.shape[1], dim=1)
        out = self.pixel_shuffle(h)
        return out


class PixelShuffle(Module):
    """ND-version of PixelShuffle upscaling."""

    def __init__(self, upscale_factor: int):
        """Initialize PixelShuffle.

        Upscales spatial dimensions and decreases the channel number by reshaping.
        The first dimension is considered a batch dimension, the second dimension
        the channel dimension, and the remaining dimensions the spatial dimensions that are upscaled.

        See `mrpro.nn.PixelUnshuffle` for the inverse operation.

        Parameters
        ----------
        upscale_factor : int
            The factor by which to upscale the spatial dimensions.
        """
        super().__init__()
        self.upscale_factor = upscale_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale the input.

        Parameters
        ----------
        x
            Tensor of shape `batch, channels, *spatial_dims`

        Returns
        -------
        Tensor of shape `batch, channels / upscale_factor**dim, *spatial_dims * upscale_factor`
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale the input."""
        dim = x.ndim - 2
        if dim == 2:  # fast path for 2D
            return torch.nn.functional.pixel_shuffle(x, self.upscale_factor)

        new_shape = (x.shape[0], -1, *(old * self.upscale_factor for old in x.shape[-dim:]))

        x = x.unflatten(1, (-1, *(self.upscale_factor,) * dim))
        x = x.moveaxis(tuple(range(2, 2 + dim)), tuple(range(-2 * dim + 1, 0, 2)))
        x = x.reshape(new_shape)
        return x
