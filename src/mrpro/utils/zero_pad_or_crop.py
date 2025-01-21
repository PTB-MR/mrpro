"""Zero pad and crop data tensor."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812


def normalize_index(ndim: int, index: int) -> int:
    """Normalize possibly negative indices.

    Parameters
    ----------
    ndim
        number of dimensions
    index
        index to normalize. negative indices count from the end.

    Raises
    ------
    IndexError
        if index is outside [-ndim,ndim)
    """
    if 0 < index < ndim:
        return index
    elif -ndim <= index < 0:
        return ndim + index
    else:
        raise IndexError(f'Invalid index {index} for {ndim} data dimensions')


def zero_pad_or_crop(
    data: torch.Tensor,
    new_shape: Sequence[int] | torch.Size,
    dim: None | Sequence[int] = None,
) -> torch.Tensor:
    """Change shape of data by center cropping or symmetric zero-padding.

    Parameters
    ----------
    data
        data
    new_shape
        desired shape of data
    dim:
        dimensions the new_shape corresponds to. None (default) is interpreted as last len(new_shape) dimensions.

    Returns
    -------
        data zero padded or cropped to shape
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
        data = F.pad(data, npad[::-1])
    return data
