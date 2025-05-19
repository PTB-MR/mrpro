"""RMSNorm module for root mean square normalization."""

import torch
from torch.nn import Module, Parameter


class RMSNorm(Module):
    """RMSNorm over the channel dimension."""

    def __init__(self, channels: int, eps: float = 1e-8):
        """Initialize RMSNorm.

        Includes a learnable weight and bias.

        Parameters
        ----------
        channels
            Number of channels.
        eps
            Epsilon value to avoid division by zero.
        """
        super().__init__()
        self.weight = Parameter(torch.zeros(channels))
        self.bias = Parameter(torch.zeros(channels))
        self.eps = eps

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
        mean_square = x.pow(2).mean(dim=1, keepdim=True)
        scale = (mean_square + self.eps).rsqrt()
        x = x * scale
        shape = (1, -1, *([1] * (x.ndim - 2)))
        weight = (1 + self.weight).view(shape)
        bias = self.bias.view(shape)
        return x * weight + bias
