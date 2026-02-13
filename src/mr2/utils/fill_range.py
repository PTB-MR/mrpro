"""Fill tensor in-place along a specified dimension with increasing integers."""

import torch

from mr2.utils.reshape import normalize_index


def fill_range_(tensor: torch.Tensor, dim: int) -> None:
    """
    Fill tensor in-place along a specified dimension with increasing integers.

    Parameters
    ----------
    tensor
        The tensor to be modified in-place.

    dim
        The dimension along which to fill with increasing values.
    """
    dim = normalize_index(tensor.ndim, dim)
    shape = [s if d == dim else 1 for d, s in enumerate(tensor.shape)]
    values = torch.arange(tensor.size(dim), device=tensor.device).reshape(shape)
    tensor[:] = values.expand_as(tensor)
