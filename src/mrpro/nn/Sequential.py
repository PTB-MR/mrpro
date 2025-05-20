"""Sequential container with support for conditioning and Operators."""

from collections import OrderedDict

import torch

from mrpro.nn.CondMixin import CondMixin
from mrpro.operators import Operator


class Sequential(torch.nn.Sequential):
    """Sequential container with support for conditioning and Operators."""

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply all modules in series to the input.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The (optional) conditioning tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply all modules in series to the input."""
        for module in self:
            if isinstance(module, CondMixin):
                x = module(x, cond)
            elif isinstance(module, Operator):
                (x,) = module(x)
            else:
                x = module(x)
        return x

    def __getitem__(self, idx: slice | int) -> 'Sequential':
        """Get a slice or item from the Sequential container.

        Subclasses will decompose to `Sequential` on indexing.
        """
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)
