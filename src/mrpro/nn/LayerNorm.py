"""Layer normalization."""

import torch
from torch.nn import Module, Parameter

from mrpro.utils.reshape import unsqueeze_right


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(self, channels: int | None, features_last: bool = False, bias: bool = True) -> None:
        """Initialize the layer normalization.

        Parameters
        ----------
        channels
            Number of channels in the input tensor. If `None`, the layer normalization does not do an elementwise
            affine transformation.
        features_last
            If `True`, the channel dimension is the last dimension.
        bias
            If `False`, only a scaling is applied without an offset if an affine transformation is used.
        """
        super().__init__()
        if channels is not None:
            self.weight: Parameter | None = Parameter(torch.ones(channels))
            self.bias: Parameter | None = Parameter(torch.zeros(channels)) if bias else None
        else:
            self.weight = None
            self.bias = None
        self.features_last = features_last

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
            Normalized output tensor
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input tensor."""
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / (std + 1e-5)

        if self.weight is not None:
            if self.features_last:
                x = x * self.weight
            else:
                x = x * unsqueeze_right(self.weight, x.ndim - 2)

        if self.bias is not None:
            if self.features_last:
                x = x + self.bias
            else:
                x = x + unsqueeze_right(self.bias, x.ndim - 2)

        return x
