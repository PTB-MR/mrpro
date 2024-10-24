"""Tensor reshaping utilities."""

from collections.abc import Sequence

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
    """Replace expanded dimensions in a view by singletons.

    Reduce either all or a specific dimension to a singleton only if it
    points to the same memory address.
    This undoes expand.

    Parameters
    ----------
    x
        input tensor
    dim
        only undo expands in the specified dimensions.
        If None, undo expand in all dimensions.
    """
    if dim is None:
        dim_: Sequence[int] = range(x.ndim)
    elif isinstance(dim, Sequence):
        dim_ = [d % x.ndim for d in dim]
    else:
        dim_ = [dim % x.ndim]

    stride = x.stride()
    size = [
        1 if stride == 0 and d in dim_ else size for d, (size, stride) in enumerate(zip(x.size(), stride, strict=True))
    ]
    return torch.as_strided(x, size, stride)
