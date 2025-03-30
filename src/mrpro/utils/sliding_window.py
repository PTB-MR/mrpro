"""Efficient sliding window implementation for tensors."""

from collections.abc import Sequence

import torch


def sliding_window(
    x: torch.Tensor,
    window_shape: int | Sequence[int],
    dim: int | Sequence[int] | None = None,
    stride: int | Sequence[int] | None = 1,
    dilation: int | Sequence[int] = 1,
) -> torch.Tensor:
    """Create a sliding window view into a tensor.

    Returns a view into tensor x where new axes representing the number of windows
    are added at the front, and the original axes involved in the sliding operation
    are replaced by window dimensions.

    Example:
        Input shape (D1, D2, D3, D4, D5), dim=(1, 3), window_shape=(k2, k4)
        Output shape: (n_windows_2, n_windows_4, D1, k2, D3, k4, D5)

    Parameters
    ----------
    x
        Input tensor to create sliding windows from
    window_shape
        Size of window over each dimension
    dim
        Dimension(s) to apply to. If None, applies to all dimensions.
    stride
        Stride of the sliding window. If None, equals window_shape.
    dilation
        Spacing between window elements for each dimension.

    Returns
    -------
    A view of the tensor with window dimensions at the front and
    original sliding axes replaced by kernel dimensions.
    """
    ndim = x.ndim

    if dim is None:
        axes = tuple(range(ndim))
    elif isinstance(dim, int):
        axes = (dim % ndim,)
    else:
        axes = tuple(ax % ndim for ax in dim)
        if len(set(axes)) != len(axes):
            raise ValueError('Duplicate values in axis are not allowed')

    n_axes = len(axes)
    window_sizes = (window_shape,) * n_axes if isinstance(window_shape, int) else tuple(window_shape)
    strides = window_sizes if stride is None else ((stride,) * n_axes if isinstance(stride, int) else tuple(stride))
    dilations = (dilation,) * n_axes if isinstance(dilation, int) else tuple(dilation)

    if any(len(param) != n_axes for param in [window_sizes, strides, dilations]):
        raise ValueError(f'Length mismatch: window_shape, stride, and dilation must all have length {n_axes}')

    if any(w <= 0 for w in window_sizes):
        raise ValueError('window_shape must be positive')
    if any(s <= 0 for s in strides):
        raise ValueError('strides must be positive')
    if any(d <= 0 for d in dilations):
        raise ValueError('dilation must be positive')

    effective_sizes = [(d * (w - 1) + 1) for w, d in zip(window_sizes, dilations, strict=False)]
    axis_to_idx = {ax: i for i, ax in enumerate(axes)}
    out_shape = []
    for i, ax in enumerate(axes):
        n_win = (x.shape[ax] - effective_sizes[i]) // strides[i] + 1
        if n_win <= 0:
            axis_size = x.shape[ax]
            raise ValueError(
                f'Dimension {ax} with size {axis_size} is too small for '
                f'window_size={window_sizes[i]}, dilation={dilations[i]}, stride={strides[i]}'
            )
        out_shape.append(n_win)
    out_strides = [x.stride()[ax] * st for ax, st in zip(axes, strides, strict=False)]
    for i in range(ndim):
        if i in axis_to_idx:
            idx = axis_to_idx[i]
            out_shape.append(window_sizes[idx])
            out_strides.append(x.stride()[i] * dilations[idx])
        else:
            out_shape.append(x.shape[i])
            out_strides.append(x.stride()[i])

    return x.as_strided(size=out_shape, stride=out_strides)
