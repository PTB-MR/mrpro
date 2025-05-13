import torch
from torch.nn import Module

from mrpro.nn.EmbMixin import EmbMixin
from mrpro.operators import Operator


class Sequential(torch.nn.Sequential):
    """Sequential container with support for embedding and Operators."""

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Apply all modules in series to the input.

        Parameters
        ----------
        x
            The input tensor.
        emb
            The (optional) embedding tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, emb)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Apply all modules in series to the input."""
        for module in self:
            if isinstance(EmbMixin, Module):
                x = module(x, emb)
            elif isinstance(module, Operator):
                (x,) = module(x)
            else:
                x = module(x)
        return x
