"""reduce_repeat utility function."""

from collections.abc import Sequence

import torch


def reduce_repeat(tensor: torch.Tensor, tol: float = 1e-6, dim: Sequence[int] | None = None) -> torch.Tensor:
    """Replace dimensions with all equal values with singletons.

    Parameters
    ----------
    tensor
        Input tensor
    tol
        tolerance.
    dim
        dimensions to try to reduce to singletons. `None` means all.
    """
    if tensor.is_complex():
        real = reduce_repeat(tensor.real, tol, dim)
        imag = reduce_repeat(tensor.imag, tol, dim)
        return real + 1j * imag

    if dim is not None:
        dim = [d % tensor.ndim for d in dim]

    def can_be_singleton(d: int) -> bool:
        if dim is not None and d not in dim:
            # not in the list of dimensions to reduce
            return False
        if tensor.stride(d) == 0:
            # broadcasted dimension
            return True
        # If the distance between min and max is smaller than the tolerance, all values are the same.
        return bool(torch.all((tensor.amax(dim=d) - tensor.amin(dim=d)) <= tol).item())

    take_first = slice(0, 1)
    take_all = slice(None)
    index = tuple(take_first if can_be_singleton(dim) else take_all for dim in range(tensor.ndim))
    return tensor[index]
