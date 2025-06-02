from typing import Literal

import torch
from torch.nn import Module


class Upsample(Module):
    def __init__(self, dim: int, scale_factor: int = 2, mode: Literal['nearest', 'linear', 'cubic'] = 'linear'):
        """Initialize the upsampling layer.

        Parameters
        ----------
        dim
            Spatial dimensions of the input tensor, i.e. 2 for 2D, 3 for 3D, etc.
        scale_factor
            Factor by which to upsample
        mode
            Interpolation mode. See `torch.nn.functional.interpolate` for details.
        """
        super().__init__()
        self.scale_factor = scale_factor
        if mode == 'nearest':
            self.mode = 'nearest'
        elif dim == 1 and mode == 'linear':
            self.mode = 'linear'
        elif dim == 2 and mode == 'cubic':
            self.mode = 'bicubic'
        elif dim == 2 and mode == 'linear':
            self.mode = 'bilinear'
        elif dim == 3 and mode == 'linear':
            self.mode = 'trilinear'
        else:
            raise ValueError(f'Invalid mode for dimension {dim}: {mode}')

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
