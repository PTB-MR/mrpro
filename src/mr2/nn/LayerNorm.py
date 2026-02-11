"""Layer normalization."""

import torch
from torch.nn import Linear, Module, Parameter

from mr2.nn.CondMixin import CondMixin
from mr2.utils.reshape import unsqueeze_at, unsqueeze_right


class LayerNorm(CondMixin, Module):
    """Layer normalization."""

    def __init__(self, n_channels: int | None, features_last: bool = False, cond_dim: int = 0) -> None:
        """Initialize the layer normalization.

        Parameters
        ----------
        n_channels
            Number of channels in the input tensor. If `None`, the layer normalization does not do an elementwise
            affine transformation.
        features_last
            If `True`, the channel dimension is the last dimension.
        cond_dim
            Number of channels in the conditioning tensor. If `0`, no adaptive scaling is applied.
        """
        super().__init__()
        if n_channels is None and cond_dim == 0:
            self.weight: Parameter | None = None
            self.bias: Parameter | None = None
            self.cond_proj: Linear | None = None
        elif n_channels is None and cond_dim > 0:
            raise ValueError('channels must be provided if cond_dim > 0')
        elif n_channels is not None and cond_dim == 0:
            self.weight = Parameter(torch.ones(n_channels))
            self.bias = Parameter(torch.zeros(n_channels))
            self.cond_proj = None
        elif n_channels is not None:
            self.weight = None
            self.bias = None
            self.cond_proj = Linear(cond_dim, 2 * n_channels)
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
        dim = -1 if self.features_last else 1
        dtype = x.dtype
        x = x.float()
        var, mean = torch.var_mean(x, dim=dim, unbiased=False, keepdim=True)
        x = (x - mean) * (var + 1e-5).rsqrt()
        x = x.to(dtype)

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
