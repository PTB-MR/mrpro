import torch
from einops import rearrange
from torch.nn import Module

from mrpro.nn.EmbMixin import EmbMixin


class ComplexAsChannel(EmbMixin, Module):
    """Wrap module to treat complex numbers as a channel dimension."""

    def __init__(self, module: Module):
        """Initialize the ComplexAsChannel module.

        Wraps a module to treat complex numbers as a channel dimension.
        If called with a complex tensor, real and imaginary parts are concatenated along the channel dimension.
        as ``(batch, (channel real/imaginary), ...)``.


        Parameters
        ----------
        module : Module
            The module to wrap.
        """
        super().__init__()
        self.module = module

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        emb : torch.Tensor | None
            The embedding tensor.
        """
        return super().__call__(x, emb)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module."""
        if x.is_complex():
            x_real = torch.view_as_real(x)
            x_real = rearrange(x_real, 'batch channel ... complex -> batch (channel complex) ...')
        else:
            x_real = x

        y = self.module(x_real)

        if x.is_complex():
            y = rearrange(y, 'b (c x y) 2 -> batch channel ... complex', complex=2).contiguous()
            y = torch.view_as_complex(y)
        return y
