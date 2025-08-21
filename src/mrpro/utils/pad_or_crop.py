"""Zero pad and crop data tensor."""

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.utils.indexing import normalize_index


def pad_or_crop(
    data: torch.Tensor,
    new_shape: Sequence[int] | torch.Size,
    dim: None | Sequence[int] = None,
    mode: Literal['constant', 'replicate', 'circular'] = 'constant',
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
        Mode for padding.
    value
        Value to use for padding.

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

    npad = []
    for old, new in zip(data.shape, new_shape, strict=True):
        diff = new - old
        after = math.trunc(diff / 2)
        before = diff - after
        npad.append(before)
        npad.append(after)

    if any(npad):
        # F.pad expects paddings in reversed order
        data = F.pad(data, npad[::-1], mode=mode, value=value)
    return data
