"""Base class for modules using a conditioning."""

import torch
from torch.nn import Module


def call_with_cond(module: Module, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
    """Call a module with conditioning if it is a CondMixin."""
    if isinstance(module, CondMixin):
        return module(*x, cond=cond)
    return module(*x)


class CondMixin(Module):
    """Mixin for modules using a conditioning.

    Used to determine if a module uses a conditioning within a Sequential container.
    """

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module to the input."""
        return super().__call__(x, cond=cond)
