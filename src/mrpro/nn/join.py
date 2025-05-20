from typing import Literal, Sequence

import torch
from torch.nn import Module

from mrpro.utils.pad_or_crop import pad_or_crop


def fix_shapes(
    xs: Sequence[torch.Tensor], mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular'], dim: Sequence[int]
) -> tuple[torch.Tensor, ...]:
    if mode == 'fail':
        return tuple(xs)

    shapes = [[x.shape[d] for d in dim] for x in xs]
    if mode == 'crop':
        target = tuple(min(s) for s in zip(*shapes, strict=True))
    else:
        target = tuple(max(s) for s in zip(*shapes, strict=True))
    if mode in ('crop', 'zero'):
        mode = 'constant'
    return tuple(pad_or_crop(x, target, dim=dim, mode=mode) for x in xs)

    # # def pad(x) -> torch.Tensor:
    # #     if x.shape[2:] == target:
    # #         return x
    # #     pad = []
    # #     for cur, tgt in zip(reversed(x.shape[2:]), reversed(target), strict=True):
    # #         left = (tgt - cur) // 2
    # #         right = tgt - cur - left
    # #         pad.extend([left, right])
    # #     return torch.nn.functional.pad(x, pad, mode=mode)

    # return tuple(pad(x) for x in xs)


class Concat(Module):
    """Concatenate tensors along the channel dimension"""

    def __init__(self, mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular'] = 'zero', dim: int = 1) -> None:
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
        xs = fix_shapes(xs, self.mode, dim=[i for i in range(max(x.ndim for x in xs)) if i != self.dim])
        return torch.cat(xs, dim=1)


class Add(Module):
    """Add tensors"""

    def __init__(self, mode: Literal['fail', 'crop', 'zero', 'replicate', 'circular'] = 'zero') -> None:
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
        xs = fix_shapes(xs, self.mode, dim=range(max(x.ndim for x in xs)))
        return sum(xs, start=torch.tensor(0.0))
