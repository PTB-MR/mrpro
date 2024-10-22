"""remove_repeat utility function."""

import torch


def remove_repeat(tensor: torch.Tensor, tol: float) -> torch.Tensor:
    """Replace dimensions with all equal values with singletons.

    Parameters
    ----------
    tensor:
        input tensor. Must be real
    tol:
        tolerance
    """

    def can_be_singleton(dim: int) -> bool:
        # If the distance between min and max is smaller than the tolerance, all values are the same.
        return bool(torch.all((tensor.amax(dim=dim) - tensor.amin(dim=dim)) <= tol).item())

    take_first = slice(0, 1)
    take_all = slice(None)
    index = tuple(take_first if can_be_singleton(dim) else take_all for dim in range(tensor.ndim))
    return tensor[index]
