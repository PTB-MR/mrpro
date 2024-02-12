"""Spatial and temporal filters."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import warnings

import numpy as np
import torch
from einops import repeat

from mrpro.data import SpatialDimension


def spatial_uniform_filter_3d(
    data: torch.Tensor, filter_width: SpatialDimension[int] | tuple[int, int, int]
) -> torch.Tensor:
    """Spatial smoothing using convolution with box function.

    Parameters
    ----------
    data
        Data to be smoothed in the shape (... z y x)
    filter_width
        Width of 3D the filter as SpatialDimension(z, y, x) or tuple(z, y, x).
        The filter width is clipped to the data shape.
    """
    # TODO: deprecate
    # Create a box-shaped filter kernel
    match filter_width:
        case SpatialDimension(z, y, x) | (z, y, x):
            z = min(data.shape[-3], z)
            y = min(data.shape[-2], y)
            x = min(data.shape[-1], x)
            kernel = torch.ones(1, 1, z, y, x)
        case _:
            raise ValueError(f'Invalid filter width: {filter_width}')

    kernel /= kernel.sum()  # normalize
    kernel = kernel.to(dtype=data.dtype, device=data.device)

    # filter by 3D convolution
    # TODO: consider replacing by 3 1D convolutions for larger kernel sizes.
    reshaped = repeat(data, '... z y x -> (...) channel z y x', channel=1)  # channel dim required for conv3d
    output = torch.nn.functional.conv3d(reshaped, weight=kernel, padding='same').view_as(data)
    return output


def _filter_separable(x: torch.Tensor, kernels: list[torch.Tensor], axis: list[int]) -> torch.Tensor:
    """Apply the separable filter kernels to the tensor x along the axes axis.

    Does zero-padding to keep the output the same size as the input.

    Parameters
    ----------
    x
        Tensor to filter
    kernels
        List of 1D kernels to apply to the tensor x.
    axis
        axes to filter over. Must have the same length as kernels.
    """
    if len(axis) != len(kernels):
        raise ValueError('Must provide matching length kernels and axis arguments. ')
    if len(axis) > x.ndim:
        raise ValueError('Too many axes provided')
    for kernel, ax in zip(kernels, axis):
        x = x.moveaxis(ax, -1)
        x = torch.nn.functional.conv1d(
            x.flatten(end_dim=-2)[:, None, :], kernel[None, None, :], padding='same'
        ).reshape(x.shape)
        x = x.moveaxis(-1, ax)
    return x


def gaussian_filter(
    x: torch.Tensor,
    sigmas: float | list[float,],
    axis: int | tuple[int, ...] | None = None,
    truncate: int = 3,
) -> torch.Tensor:
    """Apply a nd-Gaussian filter.

    Parameters
    ----------
    x
        Tensor to filter
    sigmas
        Standard deviation for Gaussian kernel. If iterable, must have length equal to the number of axes.
    axis
        Axis or axes to filter over. If None, filters over all axes.
    truncate
        Truncate the filter at this many standard deviations.
    """
    sigmas = torch.tensor(sigmas) if np.iterable(sigmas) else torch.tensor([sigmas])
    if np.any(sigmas < 0):
        raise ValueError('`sigmas` cannot contain negative values')
    if axis is None:
        axis = tuple(range(x.ndim))
    if len(sigmas) != len(axis):
        raise ValueError('Must provide matching length sigmas and axis arguments. ')

    kernels = [
        torch.exp(-0.5 * (torch.arange(-truncate * sigma, truncate * sigma + 1) / sigma) ** 2) for sigma in sigmas
    ]
    kernels = [k / k.sum() for k in kernels]
    x_filtered = _filter_separable(x, kernels, axis)
    return x_filtered


def uniform_filter(
    x: torch.Tensor,
    width: float | list[float,],
    axis: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Apply a nd-uniform filter.

    Parameters
    ----------
    x
        Tensor to filter
    width
        Width of uniform kernel. If iterable, must have length equal to the number of axes.
    axis
        Axis or axes to filter over. If None, filters over all axes.
    """

    width = torch.tensor(width) if np.iterable(width) else torch.tensor([width])
    if torch.any(width % 2 != 1):
        warnings.warn('width should be odd')
    if torch.any(width < 0):
        raise ValueError('width cannot contain negative values')
    if axis is None:
        axis = tuple(range(x.ndim))
    if len(width) != len(axis):
        raise ValueError('Must provide matching length width and axis arguments. ')

    kernels = [torch.ones(width) / width for width in width]
    x_filtered = _filter_separable(x, kernels, axis)
    return x_filtered
