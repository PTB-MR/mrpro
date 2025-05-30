"""Tensor reshaping utilities."""

from collections.abc import Sequence
from functools import lru_cache
from math import prod

import einops
import torch

from mrpro.utils.typing import endomorph


def unsqueeze_right(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsqueeze multiple times in the rightmost dimension.

    Example:
        Tensor with shape `(1, 2, 3)` and `n=2` would result in a tensor with shape `(1, 2, 3, 1, 1)`.

    Parameters
    ----------
    x
        Tensor to unsqueeze.
    n
        Number of times to unsqueeze.

    Returns
    -------
        Unsqueezed tensor (view).
    """
    new_shape = list(x.shape) + [1] * n
    return x.reshape(new_shape)


def unsqueeze_left(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsqueeze multiple times in the leftmost dimension.

    Example:
        Tensor with shape `(1, 2, 3)` and `n=2` would result in a tensor with shape `(1, 1, 1, 2, 3)`.

    Parameters
    ----------
    x
        Tensor to unsqueeze.
    n
        Number of times to unsqueeze.

    Returns
    -------
        Unsqueezed tensor (view).
    """
    new_shape = [1] * n + list(x.shape)
    return x.reshape(new_shape)


def unsqueeze_at(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """Unsqueeze multiple times at a specific dimension.

    Example:
        Tensor with shape `(1, 2, 3)`, `dim=2`, and `n=2` would result in a tensor with shape `(1, 2, 1, 1, 3)`.

    Parameters
    ----------
    x
        Tensor to unsqueeze.
    dim
        Dimension to unsqueeze. Negative values are allowed.
    n
        Number of times to unsqueeze.
    """
    if n == 0:
        return x
    elif n == 1:
        return x.unsqueeze(dim)
    elif n < 0:
        raise ValueError('n must be positive')
    if not (-x.ndim - 1 <= dim <= x.ndim):
        raise IndexError(f'Dimension {dim} out of range for tensor of dimension {x.ndim}')
    if dim < 0:
        # dim=-1 should index after the last axis, etc., to match unsqueeze
        dim = x.ndim + dim + 1
    return x.reshape(*x.shape[:dim], *(n * (1,)), *x.shape[dim:])


@endomorph
def unsqueeze_tensors_at(*x, dim: int, ndim: int | None = None) -> tuple[torch.Tensor, ...]:
    """Unsqueeze tensors at a specific dimension to the same number of dimensions.

    Example:
        - Tensors with shapes `(1, 2, 3)` and `(1, 3)` and `dim=-2`
          result in tensors with shapes `(1, 2, 3)` and `(1, 1, 3)`, as the maximum number
          of input dimensions is 3.
        - Tensors with shapes `(1, 2, 3)` and `(1, 3)` and `dim=1` and `ndim=4`
          result in tensors with shapes `(1, 1, 2, 3)` and `(1, 1, 1, 3)`.

    Parameters
    ----------
    x
        Tensors to unsqueeze.
    dim
        Dimension to unsqueeze.
    ndim
        Number of dimensions to unsqueeze to. If `None`, unsqueeze to the maximum number of dimensions
        of the input tensors.

    Returns
    -------
        Unsqueezed tensors (views) with the same number of dimensions.
    """
    if ndim is None:
        ndim_ = max(el.ndim for el in x)
    elif ndim < min(el.ndim for el in x):
        raise ValueError('ndim must be greater or equal to the minimum number of dimensions of the input tensors')
    else:
        ndim_ = ndim
    return tuple(unsqueeze_at(el, dim, n=ndim_ - el.ndim) for el in x)


@endomorph
def unsqueeze_tensors_left(*x: torch.Tensor, ndim: int | None = None) -> tuple[torch.Tensor, ...]:
    """Unsqueeze tensors on the left to the same number of dimensions.

    Parameters
    ----------
    x
        Tensors to unsqueeze.
    ndim
        Minimum number of dimensions to unsqueeze to. If `None`, unsqueeze to the maximum number of dimensions
        of the input tensors.

    Returns
    -------
        Unsqueezed tensors (views).
    """
    ndim_ = max(el.ndim for el in x)
    if ndim is not None:
        ndim_ = max(ndim_, ndim)
    return tuple(unsqueeze_left(el, ndim_ - el.ndim) for el in x)


@endomorph
def unsqueeze_tensors_right(*x: torch.Tensor, ndim: int | None = None) -> tuple[torch.Tensor, ...]:
    """Unsqueeze tensors on the right to the same number of dimensions.

    Parameters
    ----------
    x
        Tensors to unsqueeze.
    ndim
        Minimum number of dimensions to unsqueeze to. If `None`, unsqueeze to the maximum number of dimensions
        of the input tensors.

    Returns
    -------
        Unsqueezed tensors (views).
    """
    ndim_ = max(el.ndim for el in x)
    if ndim is not None:
        ndim_ = max(ndim_, ndim)
    return tuple(unsqueeze_right(el, ndim_ - el.ndim) for el in x)


@endomorph
def broadcast_right(*x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcasting on the right.

    Given multiple tensors, apply broadcasting with unsqueezed on the right.
    First, tensors are unsqueezed on the right to the same number of dimensions.
    Then, `torch.broadcast_tensors` is used.

    ```{note}
    `broadcast_left` is regular `torch.broadcast_tensors`.
    ```

    Example:
        Tensors with shapes `(1, 2, 3)`, `(1, 2)`, `(2)` result in tensors with shapes `(2, 2, 3)`.

    Parameters
    ----------
    x
        Tensors to broadcast.

    Returns
    -------
        Broadcasted tensors (views).
    """
    max_dim = max(el.ndim for el in x)
    unsqueezed = torch.broadcast_tensors(*(unsqueeze_right(el, max_dim - el.ndim) for el in x))
    return unsqueezed


def reduce_view(x: torch.Tensor, dim: int | Sequence[int] | None = None) -> torch.Tensor:
    """Reduce expanded dimensions in a view to singletons.

    Reduce either all or specific dimensions to a singleton if it
    points to the same memory address.
    This undoes `torch.Tensor.expand`.

    Parameters
    ----------
    x
        Input tensor.
    dim
        Only reduce expanded dimensions in the specified dimensions.
        If `None`, reduce all expanded dimensions.
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
    """Get reshape reduce index (cached helper function for `reshape_broadcasted`).

    This function tries to group axes from new_shape and old_shape into the smallest groups that have
    the same number of elements, starting from the right.
    If all axes of old shape of a group are stride=0 dimensions, we can reduce them.

    Example:
        old_shape = (30, 2, 2, 3)
        new_shape = `(6, 5, 4, 3)`
        Will result in the groups (starting from the right):
            - old: 3     new: 3
            - old: 2, 2  new: 4
            - old: 30    new: 6, 5
        Only the "old" groups are important.
        If all axes that are grouped together in an "old" group are stride 0 (=broadcasted),
        we can collapse them to singleton dimensions.

    This function returns the indexer that either collapses dimensions to singleton or keeps all
    elements, i.e., the slices in the returned list are all either slice(1) or slice(None).
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
                # we don't need to track the new group, just the number of elements covered.
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
        The target shape for the tensor. One of the values can be ``-1`` and its size will be inferred.

    Returns
    -------
        A tensor reshaped to the target shape, preserving broadcasted dimensions where feasible.
    """
    try:
        # if we can view the tensor directly, it will preserve broadcasting
        return tensor.view(shape)
    except RuntimeError:
        # we cannot do a view, we need to do more work:

        # -1 means infer size, i.e., the remaining elements of the input not already covered by the other axes.
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
            known_size = -prod(shape)  # negative, as it includes the -1
            if size % known_size:
                # non-integer result. no possible size of the -1 axis exists.
                raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}") from None
            shape = tuple(size // known_size if s == -1 else s for s in shape)

        # most of the broadcasted dimensions can be preserved: only dimensions that are joined with non-
        # broadcasted dimensions cannot be preserved and must be made contiguous.
        # all dimensions that can be preserved as broadcasted are first collapsed to singleton,
        # such that contiguous does not create copies along these axes.
        idx = _reshape_idx(tensor.shape, shape, tensor.stride())
        # make contiguous only in dimensions in which broadcasting cannot be preserved
        semicontiguous = tensor[idx].contiguous()
        # finally, we can expand the broadcasted dimensions to the requested shape
        semicontiguous = semicontiguous.expand(tensor.shape)
        return semicontiguous.view(shape)


def ravel_multi_index(multi_index: Sequence[torch.Tensor], dims: Sequence[int]) -> torch.Tensor:
    """
    Convert a multi-dimensional index into a flat index.

    Parameters
    ----------
    multi_index
        Sequence of integer index tensors.
    dims
        The shape of the tensor being indexed.

    Returns
    -------
    index
        Flattened index.
    """
    flat_index = multi_index[0]
    for idx, dim in zip(multi_index[1:], dims[1:], strict=True):
        flat_index = flat_index * dim + idx
    return flat_index


def broadcasted_rearrange(
    tensor: torch.Tensor,
    pattern: str,
    broadcasted_shape: Sequence[int] | None = None,
    *,
    reduce_views: bool = True,
    **axes_lengths: int,
) -> torch.Tensor:
    """Rearrange a tensor with broadcasting.

    Performs the einops rearrange or repeat operation on a tensor while preserving broadcasting.

    Rearranging is a smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation),
    reshape (view), squeeze, unsqueeze, repeat, and tile functions.

    If a tensor has stride-0 dimensions, by default they will be preserved as stride-0
    if possible and not made contiguous, thus saving memory.
    If `reduce_views` is True, then stride-0 dimensions will be reduced to singleton dimensions after rearranging.
    Optionally performs broadcasting to a specified shape before rearranging.

    Examples
    --------
    ```python
    >>> tensor = torch.randn(1, 16, 1, 768, 256)
    >>> broadcasted_rearrange(tensor, '... (phase k1) k0 -> phase ... k1 k0', phase=8, reduce_views=False).shape
    torch.Size([8, 1, 16, 1, 96, 256])

    >>> tensor=torch.randn(1, 1, 1, 768, 1)
    >>> broadcasted_rearrange(tensor, '... (phase k1) k0 -> phase ... k1 k0',
    >>>    broadcasted_shape=(1, 16, 1, 768, 256), phase=8, reduce_views=False).shape
    torch.Size([8, 1, 16, 1, 96, 256]) # Behaves as-if the tensor was of shape (1, 16, 1, 768, 256)

    >>> tensor=torch.randn(1, 1, 1, 768, 1)
    >>> broadcasted_rearrange(tensor, '... (phase k1) k0 -> phase ... k1 k0',
    >>>    broadcasted_shape=(1, 16, 1, 768, 256) phase=8, reduce_views=True).shape
    torch.Size([8, 1, 1, 1, 96, 1]) # Dimensions that are stride-0 are reduced to singleton dimensions
    ```

    Parameters
    ----------
    tensor
        The input tensor to rearrange.
    pattern
        The rearrange pattern. See `einops` documentation for more information.
    broadcasted_shape
        The shape to broadcast the tensor to before rearranging. If `None`, no additional broadcasting is performed.
    reduce_views
        If `True`, reduce stride-0 dimensions to singleton dimensions after rearranging.
    axes_lengths
        The lengths of the axes in the pattern. See `einops` documentation for more information.


    """
    tensor = tensor.broadcast_to(broadcasted_shape) if broadcasted_shape is not None else tensor
    # the broadcast-preservation is done by patching the reshape method of the tensor
    original_reshape, tensor.reshape = tensor.reshape, lambda shape: reshape_broadcasted(tensor, *shape)  # type: ignore[method-assign, assignment]
    new_tensor = einops.repeat(tensor, pattern, **axes_lengths)  # allows both repeat and rearrange
    tensor.reshape = original_reshape  # type: ignore[method-assign]
    if reduce_views:
        new_tensor = reduce_view(new_tensor)
    return new_tensor


def expand_dim(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Expand a tensor in one dimension.

    Parameters
    ----------
    tensor
        The tensor to expand.
    dim
        The dimension to expand.
    size
        The size to expand to.
    """
    new_shape = list(tensor.shape)
    new_shape[dim] = size
    return tensor.expand(new_shape)


def broadcasted_concatenate(tensors: Sequence[torch.Tensor], dim: int, reduce_views: bool = True) -> torch.Tensor:
    """Concatenate tensors while preserving broadcasting.

    Parameters
    ----------
    tensors
        The tensors to concatenate.
    dim
        The dimension to concatenate along.
    reduce_views
        If `True`, reduce stride-0 dimensions to singleton dimensions after concatenating.

    Returns
    -------
        The concatenated tensor.
    """
    n_dim = tensors[0].ndim
    if any(t.ndim != n_dim for t in tensors):
        raise ValueError('All tensors must have the same number of dimensions')
    if not (-n_dim <= dim < n_dim):
        raise ValueError(f'Dimension {dim} out of range for tensor of dimension {n_dim}')
    dim = dim % n_dim

    broadcasted_shape = []
    idx = []
    for n in range(n_dim):
        if n != dim and any(t.size(n) != tensors[0].size(n) for t in tensors):
            raise ValueError('All shapes must have the same size except for the concatenation dimension')
        if n == dim:
            idx.append(slice(None))  # keep all elements
            broadcasted_shape.append(-1)
        elif all(t.stride(n) == 0 for t in tensors):
            broadcasted_shape.append(tensors[0].size(n))
            idx.append(slice(1))  # reduce to singleton
        else:
            broadcasted_shape.append(tensors[0].size(n))
            idx.append(slice(None))

    tensors = [t[idx] for t in tensors]
    result = torch.cat(tensors, dim=dim)

    if not reduce_views:  # dimensions are already reduced, we would undo this here.
        result = result.expand(broadcasted_shape)
    return result
