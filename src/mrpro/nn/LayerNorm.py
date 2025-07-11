"""Layer normalization."""

import torch
from torch.nn import Linear, Module, Parameter

from mrpro.nn.CondMixin import CondMixin
from mrpro.utils.reshape import unsqueeze_at, unsqueeze_right


class LayerNorm(CondMixin, Module):
    """Layer normalization."""

    def __init__(self, channels: int | None, features_last: bool = False, cond_dim: int = 0) -> None:
        """Initialize the layer normalization.

        Parameters
        ----------
        channels
            Number of channels in the input tensor. If `None`, the layer normalization does not do an elementwise
            affine transformation.
        features_last
            If `True`, the channel dimension is the last dimension.
        cond_dim
            Number of channels in the conditioning tensor. If `0`, no adaptive scaling is applied.
        """
        super().__init__()
        if channels is None and cond_dim == 0:
            self.weight: Parameter | None = None
            self.bias: Parameter | None = None
            self.cond_proj: Linear | None = None
        elif channels is None and cond_dim > 0:
            raise ValueError('channels must be provided if cond_dim > 0')
        elif channels is not None and cond_dim == 0:
            self.weight = Parameter(torch.ones(channels))
            self.bias = Parameter(torch.zeros(channels))
            self.cond_proj = None
        elif channels is not None:
            self.weight = None
            self.bias = None
            self.cond_proj = Linear(cond_dim, 2 * channels)
        else:
            raise ValueError('cond_dim must be zero or positive.')

        self.features_last = features_last

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply layer normalization to the input tensor.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning tensor. If `None`, no conditioning is applied.

        Returns
        -------
            Normalized output tensor
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply layer normalization to the input tensor."""
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / (std + 1e-5)

        if self.weight is not None and self.bias is not None:
            if self.features_last:
                x = x * self.weight + self.bias
            else:
                x = x * unsqueeze_right(self.weight, x.ndim - 2) + unsqueeze_right(self.bias, x.ndim - 2)

        if self.cond_proj is not None and cond is not None:
            scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
            scale = 1 + scale
            if self.features_last:
                x = x * unsqueeze_at(scale, 1, x.ndim - 2) + unsqueeze_at(shift, 1, x.ndim - 2)
            else:
                x = x * unsqueeze_right(scale, x.ndim - 2) + unsqueeze_right(shift, x.ndim - 2)

        return x
