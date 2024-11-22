"""Tensor reshaping utilities."""

from collections.abc import Sequence
from functools import lru_cache

import torch


def unsqueeze_right(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsqueeze multiple times in the rightmost dimension.

    Example:
        tensor with shape (1,2,3) and n=2 would result in tensor with shape (1,2,3,1,1)

    Parameters
    ----------
    x
        tensor to unsqueeze
    n
        number of times to unsqueeze

    Returns
    -------
    unsqueezed tensor (view)
    """
    return x.reshape(*x.shape, *(n * (1,)))


def unsqueeze_left(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsqueze multiple times in the leftmost dimension.

    Example:
        tensor with shape (1,2,3) and n=2 would result in tensor with shape (1,1,1,2,3)


    Parameters
    ----------
    x
        tensor to unsqueeze
    n
        number of times to unsqueeze

    Returns
    -------
    unsqueezed tensor (view)
    """
    return x.reshape(*(n * (1,)), *x.shape)


def broadcast_right(*x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcasting on the right.

    Given multiple tensors, apply broadcasting with unsqueezed on the right.
    First, tensors are unsqueezed on the right to the same number of dimensions.
    Then, torch.broadcasting is used.

    Example:
        tensors with shapes (1,2,3), (1,2), (2)
        results in tensors with shape (2,2,3)

    Parameters
    ----------
    x
        tensors to broadcast

    Returns
    -------
        broadcasted tensors (views)
    """
    max_dim = max(el.ndim for el in x)
    unsqueezed = torch.broadcast_tensors(*(unsqueeze_right(el, max_dim - el.ndim) for el in x))
    return unsqueezed


def reduce_view(x: torch.Tensor, dim: int | Sequence[int] | None = None) -> torch.Tensor:
    """Reduce expanded dimensions in a view to singletons.

    Reduce either all or specific dimensions to a singleton if it
    points to the same memory address.
    This undoes expand.

    Parameters
    ----------
    x
        input tensor
    dim
        only reduce expanded dimensions in the specified dimensions.
        If None, reduce all expanded dimensions.
    """
    if dim is None:
        dim_: Sequence[int] = range(x.ndim)
    elif isinstance(dim, Sequence):
        dim_ = [d % x.ndim for d in dim]
    else:
        dim_ = [dim % x.ndim]

    stride = x.stride()
    newsize = [
        1 if stride == 0 and d in dim_ else oldsize
        for d, (oldsize, stride) in enumerate(zip(x.size(), stride, strict=True))
    ]
    return torch.as_strided(x, newsize, stride)


@lru_cache
def _reshape_idx(old_shape: tuple[int, ...], new_shape: tuple[int, ...], old_stride: tuple[int, ...]) -> list[slice]:
    """Get reshape reduce index (Cached helper function for reshape_view)."""
    # This function tries to group axes from new_shape and old_shape into the smallest groups that have#
    # the same number of elements, starting from the right.
    # If all axes of old shape of a group are stride=0 dimensions,
    # we can reduce them.
    idx = []
    i, j = len(old_shape), len(new_shape)
    while i and j:
        product_new = product_old = 1
        grouped = []
        while product_old != product_new or not grouped:
            if product_old < product_new:
                i -= 1
                grouped.append(i)
                product_old *= old_shape[i]
            else:
                j -= 1
                product_new *= new_shape[j]
        # we found a group
        if all(old_stride[d] == 0 for d in grouped):
            # all dimensions are broadcasted
            # reduce to singleton
            idx.extend([slice(1)] * len(grouped))
        else:
            # preserve
            idx.extend([slice(None)] * len(grouped))

    return idx[::-1]


def reshape(tensor: torch.Tensor, *shape: int) -> torch.Tensor:
    """Reshape a tensor while preserving broadcasted (stride 0) dimensions where possible.

    Parameters
    ----------
    tensor
        The input tensor to reshape.
    shape
        The target shape for the tensor.

    Returns
    -------
        A tensor reshaped to the target shape, preserving broadcasted dimensions where feasible.

    """
    try:
        return tensor.view(shape)
    except RuntimeError:
        idx = _reshape_idx(tensor.shape, shape, tensor.stride())
        # make contiguous in all dimensions in which broadcasting cannot be preserved
        semicontiguous = tensor[idx].contiguous().expand(tensor.shape)
        return semicontiguous.view(shape)
