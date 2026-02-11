"""Modules for concatenating or adding tensors."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch.nn import Module

from mr2.utils.interpolate import interpolate
from mr2.utils.pad_or_crop import pad_or_crop


def _fix_shapes(
    xs: Sequence[torch.Tensor],
    mode: str,
    dim: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Fix shapes of input tensors by padding or cropping."""
    if mode == 'fail':
        return tuple(xs)

    shapes = [[x.shape[d] for d in dim] for x in xs]
    if mode == 'crop':  # smallest as target
        target = tuple(min(s) for s in zip(*shapes, strict=True))
    else:  # largest as target
        target = tuple(max(s) for s in zip(*shapes, strict=True))
    if mode == 'linear' or mode == 'nearest':
        return tuple(interpolate(x, target, dim=dim, mode=mode) for x in xs)  # type: ignore[arg-type]
    if mode == 'zero' or mode == 'crop':
        return tuple(pad_or_crop(x, target, dim=dim, mode='constant', value=0.0) for x in xs)
    else:
        return tuple(pad_or_crop(x, target, dim=dim, mode=mode) for x in xs)  # type: ignore[arg-type]


class Concat(Module):
    """Concatenate tensors along the channel dimension."""

    def __init__(
        self, mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular', 'linear', 'nearest'] = 'fail', dim: int = 1
    ) -> None:
        """Initialize Concat.

        Parameters
        ----------
        mode
            How to handle mismatched dimensions:
            - 'fail': do not align, raise error if shapes mismatch
            - 'crop': center-crop to smallest spatial size
            - 'zero': zero-pad to largest spatial size
            - 'replicate': pad by edge value replication
            - 'circular': circular padding
            - 'linear': linear interpolation to largest spatial size
            - 'nearest': nearest neighbor interpolation to largest spatial size
        dim
            Dimension along which to concatenate.
        """
        super().__init__()
        modes = {'fail', 'crop', 'zero', 'replicate', 'circular', 'linear', 'nearest'}
        if mode not in modes:
            raise ValueError(f'mode must be one of {modes}')
        self.mode = mode
        self.dim = dim

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Concatenate input tensors."""
        xs = _fix_shapes(xs, self.mode, dim=[i for i in range(max(x.ndim for x in xs)) if i != self.dim])
        return torch.cat(xs, dim=self.dim)

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
        mode
            How to handle mismatched dimensions:
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


class Interpolate(Module):
    """Linear interpolate between two tensors.

    As suggestions for the Hourglass Transformer [CR]_

    References
    ----------
    .. [CK] Crowson, Katherine, et al. "Scalable high-resolution pixel-space image synthesis with
        hourglass diffusion transformers." ICML 2024, https://arxiv.org/abs/2401.11605
    """

    def __init__(self, mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular'] = 'fail') -> None:
        """Initialize learned linear interpolation.

        Parameters
        ----------
        mode
            How to handle mismatched dimensions:
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
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Linear interpolate between two tensors."""
        x1, x2 = _fix_shapes((x1, x2), self.mode, dim=range(max(x.ndim for x in (x1, x2))))
        return x1 * self.weight + x2 * (1 - self.weight)

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Linear interpolate between two tensors.

        Parameters
        ----------
        x1, x2
            Input tensors

        Returns
        -------
            Interpolated tensor
        """
        return super().__call__(x1, x2)
