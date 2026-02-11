"""Absolute position encoding (APE)."""

from itertools import combinations
from math import ceil

import torch
from torch.nn import Module

from mr2.utils.reshape import unsqueeze_right


class AbsolutePositionEncoding(Module):
    """Absolute position encoding layer.

    Encodes absolute positions in a grid. Has no learnable parameters.
    """

    encoding: torch.Tensor

    def __init__(self, n_dim: int, n_features: int, include_radii: bool = True, base_resolution: int = 128):
        """Initialize absolute position encoding layer.

        Parameters
        ----------
        n_dim
            Dimensions of the input space (1, 2, or 3)
        n_features
            Number of features to encode. The input to the forward pass needs to have at least
            this many features/channels.
        include_radii
            Whether to include radius features
        base_resolution
            Base resolution for position encoding.
            Encodings are generated at this resolution and interpolated to the input shape in the forward pass.
        """
        super().__init__()

        coords = [unsqueeze_right(torch.linspace(-1, 1, base_resolution), i) for i in range(n_dim)]
        if include_radii:
            for n in range(2, n_dim + 1):
                for combination in combinations(coords, n):
                    coords.append((2 * sum([c**2 for c in combination])) ** 0.5 - 1)
        n_freqs = ceil(n_features / len(coords) / 2)
        freqs = unsqueeze_right((base_resolution) ** torch.linspace(0, 1, n_freqs), n_dim)
        encoding = []
        for coord in coords:
            encoding.append(torch.sin(coord * freqs).broadcast_to(1, -1, *((base_resolution,) * n_dim)))
            encoding.append(torch.cos(coord * freqs).broadcast_to(1, -1, *((base_resolution,) * n_dim)))
        self.register_buffer('encoding', torch.cat(encoding, dim=1)[:, :n_features])
        self.interpolation_mode = ['linear', 'bilinear', 'trilinear'][n_dim - 1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply absolute position encoding to a tensor.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Encoded tensor with absolute position information
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply absolute position encoding to a tensor."""
        features = self.encoding.shape[1]
        if features > x.shape[1]:
            raise ValueError(f'x has {x.shape[1]} features, but {features} are required')

        x_enc, x_unenc = x.split([features, x.shape[1] - features], dim=1)
        encoding = torch.nn.functional.interpolate(self.encoding, size=x_unenc.shape[2:], mode=self.interpolation_mode)
        return torch.cat((x_enc + encoding, x_unenc), dim=1)
