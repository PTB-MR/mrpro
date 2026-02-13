"""Interpolation of data tensor."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch

from mr2.utils.reshape import normalize_indices


def interpolate(
    x: torch.Tensor, size: Sequence[int], dim: Sequence[int], mode: Literal['nearest', 'linear'] = 'linear'
) -> torch.Tensor:
    """Interpolate the tensor x along the axes dim to the new size.

    Parameters
    ----------
    x
        Tensor to interpolate
    size
        New size of the tensor
    dim
        Axes to interpolate over. Must have the same length as size.
    mode
        Interpolation mode.

    Returns
    -------
        The interpolated tensor, with the new size.

    """
    if len(dim) != len(size):
        raise ValueError('Must provide matching length size and dim arguments.')

    dim = normalize_indices(x.ndim, dim)

    # return input tensor if old and new size match
    if all(x.shape[d] == s for s, d in zip(size, dim, strict=True)):
        return x

    # torch.nn.functional.interpolate only available for real tensors
    # moveaxis is not implemented for batched tensors, so vmap would fail, thus we use permute.
    x_real = torch.view_as_real(x).permute(-1, *range(x.ndim)) if x.is_complex() else x
    dim = [d + 1 for d in dim] if x.is_complex() else dim

    for s, d in zip(size, dim, strict=True):
        if s != x_real.shape[d]:
            idx = list(range(x_real.ndim))
            # swapping the last axis and the axis to filter over
            idx[d], idx[-1] = idx[-1], idx[d]
            x_real = x_real.permute(idx)
            x_real = torch.nn.functional.interpolate(x_real.flatten(end_dim=-3), size=s, mode=mode).reshape(
                *x_real.shape[:-1], -1
            )
            # for a single permutation, this undoes the permutation
            x_real = x_real.permute(idx)
    return torch.view_as_complex(x_real.permute(*range(1, x.ndim + 1), 0).contiguous()) if x.is_complex() else x_real


def apply_lowres(function: Callable[[torch.Tensor], torch.Tensor], size: Sequence[int], dim: Sequence[int]) -> Callable:
    """Apply function f on low-res version of tensor x and then return the upsampled f(x).

    Parameters
    ----------
    function
        Function to be applied on low-resolution version of tensor.
    size
        Low-resolution size of tensor.
    dim
        Low-resolution axes. Must have the same length as size.

    Returns
    -------
        Function which downsamples tensor, applies function and upsamples result.
    """

    def apply_to_lowres_data(x: torch.Tensor) -> torch.Tensor:
        """Downsample tensor, apply function and upsample result.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Tensor in original size with function applied.
        """
        x_lowres = interpolate(x, size, dim)
        x_lowres = function(x_lowres)
        return interpolate(x_lowres, [x.shape[d] for d in dim], dim)

    return apply_to_lowres_data
