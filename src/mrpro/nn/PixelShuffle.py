"""ND-version of PixelShuffle and PixelUnshuffle."""

import torch
from torch.nn import Module


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
