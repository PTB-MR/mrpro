"""Sequential container with support for conditioning and Operators."""

from collections import OrderedDict
from typing import cast

import torch

from mr2.nn.CondMixin import CondMixin
from mr2.operators import Operator


class Sequential(CondMixin, torch.nn.Sequential):
    """Sequential container with support for conditioning and Operators.

    Allows multiple input tensors and a single output tensor of the sequential block.

    """

    def __call__(self, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
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
        return torch.nn.Sequential.__call__(self, *x, cond=cond)

    def forward(self, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply all modules in series to the input."""
        for module in self:
            if isinstance(module, Operator):
                x = cast(tuple[torch.Tensor, ...], module(*x))  # always tuple
            else:
                ret: torch.Tensor | tuple[torch.Tensor, ...]
                if isinstance(module, CondMixin):
                    ret = module(*x, cond=cond)
                else:
                    ret = module(*x)
                if isinstance(ret, tuple):
                    x = ret
                else:
                    x = (ret,)
        return x[0]

    def __getitem__(self, idx: slice | int) -> 'Sequential':
        """Get a slice or item from the Sequential container.

        Subclasses will decompose to `Sequential` on indexing.
        """
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return cast(Sequential, self._get_item_by_idx(self._modules.values(), idx))
