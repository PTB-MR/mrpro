"""Upsampling by interpolation."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch.nn import Module, Sequential

from mrpro.nn.PermutedBlock import PermutedBlock


class Upsample(Module):
    """Upsampling by interpolation."""

    def __init__(
        self, dim: Sequence[int], scale_factor: int = 2, mode: Literal['nearest', 'linear', 'cubic'] = 'linear'
    ):
        """Initialize the upsampling layer.

        Parameters
        ----------
        dim
            Dimensions which to upsample
        scale_factor
            Factor by which to upsample
        mode
            Interpolation mode. See `torch.nn.functional.interpolate` for details.
        """
        super().__init__()
        self.scale_factor = scale_factor
        if mode == 'nearest':
            dims = [tuple(d) for d in torch.tensor(dim).split(3)]
            modes = ['nearest'] * len(self.dim)
        elif mode == 'linear':
            dims = [tuple(d) for d in torch.tensor(dim).split(3)]
            modes = [{1: 'linear', 2: 'bilinear', 3: 'trilinear'}[len(d)] for d in dims]
        elif mode == 'cubic':
            if not len(dim) == 2:
                raise ValueError('Cubic interpolation is only supported for 2D images.')
            dims = [tuple(dim)]
            modes = ['bicubic']

        self.blocks = Sequential(
            *[
                PermutedBlock(d, Upsample(d, scale_factor=scale_factor, mode=m))
                for d, m in zip(dims, modes, strict=False)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the input tensor."""
        return torch.nn.functional.interpolate(
            x,
            mode=self.mode,
            scale_factor=self.scale_factor,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the input tensor.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Upsampled tensor
        """
        return super().__call__(x)
