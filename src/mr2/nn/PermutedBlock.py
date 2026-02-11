"""Block that applies a submodule along selected spatial dimensions."""

from collections.abc import Sequence

import torch
from torch import nn

from mr2.nn.CondMixin import CondMixin, call_with_cond


class PermutedBlock(CondMixin, nn.Module):
    """Apply a submodule along selected spatial dimensions."""

    apply_along_dim: tuple[int, ...]
    module: nn.Module

    def __init__(self, apply_along_dim: Sequence[int], module: nn.Module, features_last: bool = False):
        """Initialize the PermutedBlock.

        Parameters
        ----------
        apply_along_dim
            Spatial dimension indices to use when applying the module.
            These will be moved to the last dimensions.
        module
            Module to apply on the selected dims.
        features_last
            If True, the features dimension is assumed to be the last dimension, as common in transformer models.
        """
        super().__init__()
        self.apply_along_dim = tuple(sorted(apply_along_dim))
        self.module = module
        self.features_last = features_last

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module along the selected dimensions.

        Parameters
        ----------
        x
            Input tensor
        cond
            Conditioning tensor, passed to the module if it supports conditioning
            (that is, if it is a subclass of `~mr2.nn.CondMixin`)

        Returns
        -------
            Output tensor.
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the module along the selected dimensions."""
        keep = tuple(d % x.ndim for d in self.apply_along_dim)
        if 0 in keep:
            raise ValueError('Batch dimension should not be in apply_along_dim.')
        if self.features_last:
            if x.ndim - 1 in keep:
                raise ValueError('Features dimension should not be in apply_along_dim.')
            batch_dim = tuple(d for d in range(1, x.ndim - 1) if d not in keep)
            permute = (0, *batch_dim, *keep, x.ndim - 1)
        else:
            if 1 in keep:
                raise ValueError('Features dimension should not be in apply_along_dim.')
            batch_dim = tuple(d for d in range(2, x.ndim) if d not in keep)
            permute = (0, *batch_dim, 1, *keep)
        h = x.permute(permute)
        batch_shape = h.shape[: 1 + len(batch_dim)]
        h = h.flatten(0, len(batch_dim))
        h = call_with_cond(self.module, h, cond=cond)
        h = h.unflatten(0, batch_shape)
        permute_back = [0] * x.ndim
        for i, p in enumerate(permute):
            permute_back[p] = i
        return h.permute(tuple(permute_back))
