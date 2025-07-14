"""ND-version of PixelShuffle and PixelUnshuffle."""

from math import ceil

import torch
from torch.nn import Module

from mrpro.nn.ndmodules import ConvND


class PixelUnshuffle(Module):
    """ND-version of PixelUnshuffle downscaling."""

    def __init__(self, downscale_factor: int, features_last: bool = False):
        """Initialize PixelUnshuffle.

        Reduces spatial dimensions and increases the channel number by reshaping.
        The first dimension is considered a batch dimension, the second dimension
        the channel dimension, and the remaining dimensions the spatial dimensions that are downscaled.

        See `mrpro.nn.PixelShuffle` for the inverse operation.

        Parameters
        ----------
        downscale_factor
            The factor by which to downscale the input tensor.
        features_last
            Whether the features/channels dimension is the last dimension as common in transformer models or the
            second dimension as common in CNN models.
        """
        super().__init__()
        self.downscale_factor = downscale_factor
        self.features_last = features_last

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Downscale the input.

        Parameters
        ----------
        x
            Tensor of shape `batch, channels, *spatial_dims` or `batch, *spatial_dims, channels` (if `features_last`).

        Returns
        -------
        Tensor of shape `batch, channels * downscale_factor**dim, *spatial_dims/downscale_factor` or
        `batch, *spatial_dims/downscale_factor, channels * downscale_factor**dim` (if `features_last`).
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downscale the input."""
        n_dim = x.ndim - 2
        if n_dim == 2 and not self.features_last:  # fast path for 2D images
            return torch.nn.functional.pixel_unshuffle(x, self.downscale_factor)

        new_shape = list(x.shape[:1]) if self.features_last else list(x.shape[:2])
        source_positions = []
        for i, old in enumerate(x.shape[1:-1] if self.features_last else x.shape[2:]):
            if old % self.downscale_factor:
                raise ValueError('Spatial size must be divisible by downscale_factor.')
            new_shape.append(old // self.downscale_factor)
            new_shape.append(self.downscale_factor)
            source_positions.append(2 + 2 * i)
        if self.features_last:
            new_shape.append(x.shape[-1])
        x = x.view(new_shape)
        x = x.moveaxis(source_positions, tuple(range(-n_dim, 0)))
        if self.features_last:
            x = x.flatten(-n_dim - 1)
        else:
            x = x.flatten(1, -n_dim - 1)

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
        self, n_dim: int, n_channels_in: int, n_channels_out: int, downscale_factor: int = 2, residual: bool = False
    ):
        """Initialize a PixelUnshuffleDownsample layer.

        Parameters
        ----------
        n_dim
            Dimension of the input tensor.
        n_channels_in
            Number of channels in the input tensor.
        n_channels_out
            Number of channels in the output tensor.
        downscale_factor
            Factor by which to downscale the input tensor.
        residual
            Whether to use a residual connection as proposed in [DCAE]_.
        """
        super().__init__()
        self.pixel_unshuffle = PixelUnshuffle(downscale_factor)
        out_ratio = downscale_factor**n_dim
        if n_channels_out % out_ratio != 0:
            raise ValueError(f'channels_out must be divisible by downscale_factor**{n_dim}.')
        self.conv = ConvND(n_dim)(n_channels_in, n_channels_out // out_ratio, kernel_size=3, padding='same')
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

    def __init__(
        self, n_dim: int, n_channels_in: int, n_channels_out: int, upscale_factor: int = 2, residual: bool = False
    ):
        """Initialize a PixelShuffleUpsample layer.

        Parameters
        ----------
        n_dim
            Dimension of the input tensor.
        n_channels_in
            Number of channels in the input tensor.
        n_channels_out
            Number of channels in the output tensor.
        upscale_factor
            Factor by which to upscale the input tensor.
        residual
            Whether to use a residual connection as proposed in [DCAE]_.
        """
        super().__init__()
        self.conv = ConvND(n_dim)(n_channels_in, n_channels_out * upscale_factor**n_dim, kernel_size=3, padding='same')
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
            h = h + x.repeat_interleave(ceil(h.shape[1] / x.shape[1]), dim=1)[:, : h.shape[1]]
        out = self.pixel_shuffle(h)
        return out


class PixelShuffle(Module):
    """ND-version of PixelShuffle upscaling."""

    def __init__(self, upscale_factor: int, features_last: bool = False):
        """Initialize PixelShuffle.

        Upscales spatial dimensions and decreases the channel number by reshaping.
        The first dimension is considered a batch dimension, the second dimension
        the channel dimension, and the remaining dimensions the spatial dimensions that are upscaled.

        See `mrpro.nn.PixelUnshuffle` for the inverse operation.

        Parameters
        ----------
        upscale_factor
            The factor by which to upscale the spatial dimensions.
        features_last
            Whether the features/channels dimension is the last dimension as common in transformer models or the
            second dimension as common in CNN models.
        """
        super().__init__()
        self.upscale_factor = upscale_factor
        self.features_last = features_last

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale the input.

        Parameters
        ----------
        x
            Tensor of shape `batch, channels, *spatial_dims` or `batch, *spatial_dims, channels` (if `features_last`).

        Returns
        -------
        Tensor of shape `batch, channels / upscale_factor**n_dim, *spatial_dims * upscale_factor` or
        `batch, *spatial_dims * upscale_factor, channels / upscale_factor**n_dim` (if `features_last`).
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale the input."""
        n_dim = x.ndim - 2
        if n_dim == 2 and not self.features_last:  # fast path for 2D
            return torch.nn.functional.pixel_shuffle(x, self.upscale_factor)

        if self.features_last:
            new_shape = (x.shape[0], *(old * self.upscale_factor for old in x.shape[-n_dim - 1 : -1]), -1)
            x = x.unflatten(-1, (-1, *(self.upscale_factor,) * n_dim))
            x = x.moveaxis(tuple(range(-n_dim, 0)), tuple(range(-2 * n_dim, 0, 2)))
        else:
            new_shape = (x.shape[0], -1, *(old * self.upscale_factor for old in x.shape[-n_dim:]))
            x = x.unflatten(1, (-1, *(self.upscale_factor,) * n_dim))
            x = x.moveaxis(tuple(range(2, 2 + n_dim)), tuple(range(-2 * n_dim + 1, 0, 2)))
        x = x.reshape(new_shape)
        return x
