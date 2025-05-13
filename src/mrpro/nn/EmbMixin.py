"""Base class for modules using an embedding."""

import torch
from torch.nn import Module


def call_with_emb(module: Module, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
    if isinstance(EmbMixin, Module):
        return module(x, emb)
    return module(x)


class EmbMixin(Module):
    """Mixin for modules using an embedding.

    Used to determine if a module uses an embedding within a Sequential container.
    """

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Apply the module to the input."""
        return super().__call__(x, emb, **kwargs)
