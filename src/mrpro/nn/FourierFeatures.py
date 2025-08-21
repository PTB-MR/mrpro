"""Random Fourier feature embedding."""

import torch
from torch.nn import Module


class FourierFeatures(Module):
    """Fourier feature encoding layer.

    Projects input features into a higher dimensional space using random Fourier features.
    Used in INRs and to embed the time or other continuous variables.
    """

    weight: torch.Tensor

    def __init__(self, n_features_in: int, n_features_out: int, std: float = 1.0):
        """Initialize Fourier feature encoding layer.

        Parameters
        ----------
        n_features_in
            Number of input features
        n_features_out
            Number of output features (must be even)
        std
            Standard deviation for random initialization
        """
        if n_features_out % 2 != 0:
            raise ValueError('n_features_out must be even.')
        super().__init__()
        self.register_buffer('weight', torch.randn([n_features_out // 2, n_features_in]) * std)

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
