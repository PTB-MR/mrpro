"""Zero pad and crop data tensor."""

import math
from collections.abc import Sequence
from typing import Literal

import torch

from mr2.utils.reshape import normalize_index, unsqueeze_left


def pad_or_crop(
    data: torch.Tensor,
    new_shape: Sequence[int] | torch.Size,
    dim: None | Sequence[int] = None,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0,
) -> torch.Tensor:
    """Change shape of data by center cropping or symmetric padding.

    Parameters
    ----------
    data
        Data to pad or crop.
    new_shape
        Desired shape of data.
    dim
        Dimensions the `new_shape` corresponds to.
        `None` is interpreted as last ``len(new_shape)`` dimensions.
    mode
        Mode to use for padding.
    value
        Value to use for constant padding.

    Returns
    -------
        Data zero padded or cropped to shape.
    """
    if len(new_shape) > data.ndim:
        raise ValueError('length of new shape should not exceed dimensions of data')

    if dim is None:  # Use last dimensions
        new_shape = (*data.shape[: -len(new_shape)], *new_shape)
    else:
        if len(new_shape) != len(dim):
            raise ValueError('length of shape should match length of dim')
        dim = tuple(normalize_index(data.ndim, idx) for idx in dim)  # raises if any not in [-data.ndim,data.ndim)
        if len(dim) != len(set(dim)):  # this is why we normalize
            raise ValueError('repeated values are not allowed in dims')
        # Update elements in data.shape at indices specified in dim with corresponding elements from new_shape
        new_shape = tuple(new_shape[dim.index(i)] if i in dim else s for i, s in enumerate(data.shape))

    npad: list[int] = []
    for old, new in zip(data.shape, new_shape, strict=True):
        diff = new - old
        after = math.trunc(diff / 2)
        before = diff - after
        if before or after or npad:
            npad.append(before)
            npad.append(after)

    n_extended_dims = 0
    if mode != 'constant':
        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html for supported shapes.
        while len(npad) // 2 < data.ndim - 2:
            npad = [0, 0, *npad]

        n_extended_dims = max(0, len(npad) // 2 - (data.ndim - 2))
        if n_extended_dims:  # We need to extend data such that the padding is supported.
            data = unsqueeze_left(data, n_extended_dims)

        if len(npad) > 6:  # TODO: reshape and call multiple times
            raise ValueError('Non-constant padding is only supported for up to the last 3 dimensions.')

    if any(npad):
        # F.pad expects paddings in reversed order
        data = torch.nn.functional.pad(data, npad[::-1], value=value, mode=mode)

    if n_extended_dims:
        idx = n_extended_dims * (0,)
        data = data[idx]
    return data
