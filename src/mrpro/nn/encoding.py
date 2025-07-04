"""Encoding modules for neural networks."""

from itertools import combinations
from math import ceil

import torch
from torch.nn import Module

from mrpro.utils.reshape import unsqueeze_right


class FourierFeatures(Module):
    """Fourier feature encoding layer.

    Projects input features into a higher dimensional space using random Fourier features.
    This is useful for encoding positional information in neural networks.
    """

    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        """Initialize Fourier feature encoding layer.

        Parameters
        ----------
        in_features
            Number of input features
        out_features
            Number of output features (must be even)
        std
            Standard deviation for random initialization
        """
        if out_features % 2 != 0:
            raise ValueError('out_features must be even.')
        super().__init__()
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding.

        Parameters
        ----------
        x
            Input tensor of shape (..., in_features)

        Returns
        -------
        Encoded features of shape (..., out_features)
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding."""
        f = 2 * torch.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class AbsolutePositionEncoding(Module):
    """Absolute position encoding layer.

    Encodes absolute positions in a grid using learned embeddings.
    """

    encoding: torch.Tensor

    def __init__(self, dim: int, features: int, include_radii: bool = True, base_resolution: int = 128):
        """Initialize absolute position encoding layer.

        Parameters
        ----------
        dim
            Dimension of the input space (1, 2, or 3)
        features
            Number of output features
        include_radii
            Whether to include radius features
        base_resolution
            Base resolution for position encoding
        """
        super().__init__()

        coords = [unsqueeze_right(torch.linspace(-1, 1, base_resolution), i) for i in range(dim)]
        if include_radii:
            for n in range(2, dim + 1):
                for combination in combinations(coords, n):
                    coords.append((2 * sum([c**2 for c in combination])) ** 0.5 - 1)
        n_freqs = ceil(features / len(coords) / 2)
        freqs = unsqueeze_right((base_resolution) ** torch.linspace(0, 1, n_freqs), dim)
        encoding = []
        for coord in coords:
            encoding.append(torch.sin(coord * freqs).broadcast_to(1, -1, *((base_resolution,) * dim)))
            encoding.append(torch.cos(coord * freqs).broadcast_to(1, -1, *((base_resolution,) * dim)))
        self.register_buffer('encoding', torch.cat(encoding, dim=1)[:, :features])
        self.interpolation_mode = ['linear', 'bilinear', 'trilinear'][dim - 1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for encoding.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
        Encoded tensor with absolute position information
        """
        features = self.encoding.shape[1]
        if features > x.shape[1]:
            raise ValueError(f'x has {x.shape[1]} features, but {features} are required')

        x_enc, x_unenc = x.split([features, x.shape[1] - features], dim=1)
        encoding = torch.nn.functional.interpolate(self.encoding, size=x_unenc.shape[2:], mode=self.interpolation_mode)
        return torch.cat((x_enc + encoding, x_unenc), dim=1)
