"""Sliding window view."""

import warnings
from collections.abc import Sequence

import torch


def sliding_window(
    x: torch.Tensor,
    window_shape: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
    strides: int | Sequence[int] = 1,
) -> torch.Tensor:
    """Sliding window view into the tensor x.

    Returns a view into the tensor x that represents a sliding window.
    The window-axes will be at the end in the order of the axis argument.
    Non-overlapping windows can be achieved by setting the strides to the window_shape.
    Note that the stride argument is **experimental** and not fully supported.

    Parameters
    ----------
    x
        Tensor to slide over
    window_shape
        Size of window over each axis that takes part in the sliding window.
    axis
        Axis or axes to slide over. If None, slides over all axes.
    strides
        Stride of the sliding window. **Experimental**.
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple([ax % x.ndim for ax in axis])
        if len(set(axis)) != len(axis):
            raise ValueError('duplicate values in axis are not allowed')

    window_shape = tuple(window_shape) if isinstance(window_shape, Sequence) else (window_shape,) * len(axis)
    strides = tuple(strides) if isinstance(strides, Sequence) else (strides,) * len(axis)
    # we want to use fancy indexing, so we need these as tensors
    window_shape_tensor = torch.tensor(window_shape)
    strides_tensor = torch.tensor(strides)
    x_shape_tensor = torch.tensor(x.shape)

    if torch.any(strides_tensor != 1):
        warnings.warn('strides other than 1 are not fully supported', stacklevel=2)
    if torch.any(window_shape_tensor < 0):
        raise ValueError('window_shape cannot contain negative values')
    if torch.any(strides_tensor < 0):
        # this is a pytorch limitation. python api standard should allow negative strides
        raise ValueError('strides cannot contain negative values')
    if len(window_shape) != len(axis):
        raise ValueError('Must provide matching length window_shape and axis arguments. ')
    if len(strides) != len(axis):
        raise ValueError('Must provide matching length strides and axis arguments.')
    # out_strides should be the original strides, but for sliding windows axis the stride should be increased
    # and a new dimension should be added
    out_strides = torch.tensor([x.stride(i) for i in range(x.ndim)] + [x.stride(ax) for ax in axis])
    out_strides[axis,] = out_strides[axis,] * strides_tensor
    # remove boundaries, similar to convolution with padding="valid".
    x_shape_tensor[axis,] = (x_shape_tensor[axis,] + strides_tensor - window_shape_tensor) // strides_tensor
    if torch.any(x_shape_tensor <= 0):
        # only partial views
        raise ValueError('strides or windows too large')
    out_shape = tuple(x_shape_tensor) + window_shape
    view = x.as_strided(size=out_shape, stride=tuple(out_strides))
    return view
