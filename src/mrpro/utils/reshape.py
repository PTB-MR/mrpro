"""Tensor reshaping utilities."""

from collections.abc import Sequence
from functools import lru_cache
from math import prod

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
    """Get reshape reduce index (Cached helper function for reshape_broadcasted).

    This function tries to group axes from new_shape and old_shape into the smallest groups that have
    the same number of elements, starting from the right.
    If all axes of old shape of a group are stride=0 dimensions, we can reduce them.

    Example:
        old_shape = (30, 2, 2, 3)
        new_shape = (6, 5, 4, 3)
        Will results in the groups (starting from the right):
            - old: 3     new: 3
            - old: 2, 2  new: 4
            - old: 30    new: 6, 5
        Only the "old" groups are important.
        If all axes that are grouped together in an "old" group are stride 0 (=broadcasted)
        we can collapse them to singleton dimensions.
    This function returns the indexer that either collapses dimensions to singleton or keeps all
    elements, i.e. the slices in the returned list are all either slice(1) or slice(None).
    """
    idx = []
    pointer_old, pointer_new = len(old_shape) - 1, len(new_shape) - 1  # start from the right
    while pointer_old >= 0:
        product_new, product_old = 1, 1  # the number of elements in the current "new" and "old" group
        group: list[int] = []
        while product_old != product_new or not group:
            if product_old <= product_new:
                # increase "old" group
                product_old *= old_shape[pointer_old]
                group.append(pointer_old)
                pointer_old -= 1
            else:
                # increase "new" group
                # we don't need to track the new group, the number of elemeents covered.
                product_new *= new_shape[pointer_new]
                pointer_new -= 1
        # we found a group. now we need to decide what to do.
        if all(old_stride[d] == 0 for d in group):
            # all dimensions are broadcasted
            # -> reduce to singleton
            idx.extend([slice(1)] * len(group))
        else:
            # preserve dimension
            idx.extend([slice(None)] * len(group))
    idx = idx[::-1]  # we worked right to left, but our index should be left to right
    return idx


def reshape_broadcasted(tensor: torch.Tensor, *shape: int) -> torch.Tensor:
    """Reshape a tensor while preserving broadcasted (stride 0) dimensions where possible.

    Parameters
    ----------
    tensor
        The input tensor to reshape.
    shape
        The target shape for the tensor. One of the values can be `-1` and its size will be inferred.

    Returns
    -------
        A tensor reshaped to the target shape, preserving broadcasted dimensions where feasible.

    """
    try:
        # if we can view the tensor directly, it will preserve broadcasting
        return tensor.view(shape)
    except RuntimeError:
        # we cannot do a view, we need to do more work:

        # -1 means infer size, i.e. the remaining elements of the input not already covered by the other axes.
        negative_ones = shape.count(-1)
        size = tensor.shape.numel()
        if not negative_ones:
            if prod(shape) != size:
                # use same exception as pytorch
                raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}") from None
        elif negative_ones > 1:
            raise RuntimeError('only one dimension can be inferred') from None
        elif negative_ones == 1:
            # we need to figure out the size of the "-1" dimension
            known_size = -prod(shape)  # negative, is it includes the -1
            if size % known_size:
                # non integer result. no possible size of the -1 axis exists.
                raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}") from None
            shape = tuple(size // known_size if s == -1 else s for s in shape)

        # most of the broadcasted dimensions can be preserved: only dimensions that are joined with non
        # broadcasted dimensions can not be preserved and must be made contiguous.
        # all dimensions that can be preserved as broadcasted are first collapsed to singleton,
        # such that contiguous does not create copies along these axes.
        idx = _reshape_idx(tensor.shape, shape, tensor.stride())
        # make contiguous only in dimensions in which broadcasting cannot be preserved
        semicontiguous = tensor[idx].contiguous()
        # finally, we can expand the broadcasted dimensions to the requested shape
        semicontiguous = semicontiguous.expand(tensor.shape)
        return semicontiguous.view(shape)
