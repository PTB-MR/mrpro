"""Modules for concatenating or adding tensors."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch.nn import Module

from mrpro.utils.pad_or_crop import pad_or_crop


def _fix_shapes(
    xs: Sequence[torch.Tensor], mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular'], dim: Sequence[int]
) -> tuple[torch.Tensor, ...]:
    """Fix shapes of input tensors by padding or cropping."""
    if mode == 'fail':
        return tuple(xs)

    shapes = [[x.shape[d] for d in dim] for x in xs]
    if mode == 'crop':
        target = tuple(min(s) for s in zip(*shapes, strict=True))
    else:
        target = tuple(max(s) for s in zip(*shapes, strict=True))
    if mode == 'zero' or mode == 'crop':
        return tuple(pad_or_crop(x, target, dim=dim, mode='constant', value=0.0) for x in xs)
    else:
        return tuple(pad_or_crop(x, target, dim=dim, mode=mode) for x in xs)


class Concat(Module):
    """Concatenate tensors along the channel dimension."""

    def __init__(self, mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular'] = 'fail', dim: int = 1) -> None:
        """Initialize Concat.

        Parameters
        ----------
        mode : {'fail', 'crop', 'zero', 'replicate', 'circular'}, default='zero'
            How to handle mismatched spatial dimensions:
            - 'fail': do not align, raise error if shapes mismatch
            - 'crop': center-crop to smallest spatial size
            - 'zero': zero-pad to largest spatial size
            - 'replicate': pad by edge value replication
            - 'circular': circular padding
        dim
            Dimension along which to concatenate.
        """
        super().__init__()
        modes = {'fail', 'crop', 'zero', 'replicate', 'circular'}
        if mode not in modes:
            raise ValueError(f'mode must be one of {modes}')
        self.mode = mode
        self.dim = dim

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Concatenate input tensors."""
        xs = _fix_shapes(xs, self.mode, dim=[i for i in range(max(x.ndim for x in xs)) if i != self.dim])
        return torch.cat(xs, dim=1)

    def __call__(self, *xs: torch.Tensor) -> torch.Tensor:
        """
        Concatenate input tensors.

        Parameters
        ----------
        xs
            Input tensors

        Returns
        -------
            Concatenated tensor
        """
        return super().__call__(*xs)


class Add(Module):
    """Add tensors."""

    def __init__(self, mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular'] = 'fail') -> None:
        """Initialize Add.

        Parameters
        ----------
        mode : {'fail', 'crop', 'zero', 'replicate', 'circular'}, default='zero'
            How to handle mismatched spatial dimensions:
            - 'fail': do not align, raise error if shapes mismatch
            - 'crop': center-crop to smallest spatial size
            - 'zero': zero-pad to largest spatial size
            - 'replicate': pad by edge value replication
            - 'circular': circular padding
        """
        super().__init__()
        modes = {'fail', 'crop', 'zero', 'replicate', 'circular'}
        if mode not in modes:
            raise ValueError(f'mode must be one of {modes}')
        self.mode = mode

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Add input tensors."""
        xs = _fix_shapes(xs, self.mode, dim=range(max(x.ndim for x in xs)))
        return sum(xs, start=torch.tensor(0.0))

    def __call__(self, *xs: torch.Tensor) -> torch.Tensor:
        """
        Add input tensors.

        Parameters
        ----------
        xs
            Input tensors

        Returns
        -------
        Summed tensor
        """
        return super().__call__(*xs)
