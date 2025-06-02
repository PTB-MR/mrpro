"""RMSNorm module for root mean square normalization."""

import torch
from torch.nn import Module, Parameter


class RMSNorm(Module):
    """RMSNorm over the channel dimension."""

    def __init__(self, channels: int, eps: float = 1e-8, features_last: bool = False):
        """Initialize RMSNorm.

        Includes a learnable weight and bias.

        Parameters
        ----------
        channels
            Number of channels.
        eps
            Epsilon value to avoid division by zero.
        features_last
            If True, the channel dimension is the last dimension.
        """
        super().__init__()
        self.weight = Parameter(torch.zeros(channels))
        self.bias = Parameter(torch.zeros(channels))
        self.eps = eps
        self.channel_dim = -1 if features_last else 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm over the channel dimension.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Normalized tensor.
        """
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm over the channel dimension."""
        mean_square = x.square().mean(dim=self.channel_dim, keepdim=True)
        scale = (mean_square + self.eps).rsqrt()
        x = x * scale
        shape = [1] * x.ndim
        shape[self.channel_dim] = -1
        weight = (self.weight + 1).view(shape)
        bias = self.bias.view(shape)
        return x * weight + bias
