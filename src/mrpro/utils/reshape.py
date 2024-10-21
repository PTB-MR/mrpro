"""Tensor reshaping utilities."""

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
    """
    return x.reshape(*x.shape, *(n * (1,)))


def unsqueeze_left(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsqueze multiple times in the leftmost dimension.

    Example:
        tensor with shape (1,2,3) and n=2 would result in tensor with shape (1,1,1,1,2,3)


    Parameters
    ----------
    x
        tensor to unsqueeze
    n
        number of times to unsqueeze
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
