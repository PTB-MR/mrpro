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
