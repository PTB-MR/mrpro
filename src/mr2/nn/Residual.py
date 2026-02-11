"""Residual connection."""

import torch
from torch.nn import Identity, Module

from mr2.nn.CondMixin import CondMixin, call_with_cond


class Residual(CondMixin, Module):
    """Residual connection."""

    def __init__(self, module: Module, skip: Module | None = None):
        """Initialize the residual connection.

        Parameters
        ----------
        module
            The main path of the residual connection.
        skip
            The skip path of the residual connection. If None, the identity function is used.
        """
        super().__init__()
        self.module = module
        self.skip = Identity() if skip is None else skip

    def __call__(self, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The optional conditioning tensor. If the modules are an instance of `CondMixin`,
            the conditioning is passed to the modules.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(*x, cond=cond)

    def forward(self, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module."""
        return call_with_cond(self.module, *x, cond=cond) + call_with_cond(self.skip, *x, cond=cond)
