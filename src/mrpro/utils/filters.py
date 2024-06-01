"""Spatial and temporal filters."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from collections.abc import Sequence
from math import ceil
from typing import Literal

import numpy as np
import torch

from mrpro.data.SpatialDimension import SpatialDimension


def uniform_filter_3d(
    data: torch.Tensor,
    filter_width: SpatialDimension[int] | tuple[int, int, int] | int,
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Spatial smoothing using convolution with box function.

    Filters along the last three dimensions of the data tensor.

    Parameters
    ----------
    data
        Data to be smoothed in the shape (... z y x)
    filter_width
        Width of the filter as SpatialDimension(z, y, x) or tuple(z, y, x).
        If a single integer is supplied, it is used as the width along z, y, and x.
        The filter width is clipped to the data shape.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant
    """
    match filter_width:
        case int(width):
            z = min(data.shape[-3], width)
            y = min(data.shape[-2], width)
            x = min(data.shape[-1], width)
        case SpatialDimension(z, y, x) | (z, y, x):
            z = min(data.shape[-3], z)
            y = min(data.shape[-2], y)
            x = min(data.shape[-1], x)
        case _:
            raise ValueError(f'Invalid filter width: {filter_width}')
    return uniform_filter(data, (z, y, x), axis=(-3, -2, -1), pad_mode=pad_mode, pad_value=pad_value)


def filter_separable(
    x: torch.Tensor,
    kernels: Sequence[torch.Tensor],
    axis: Sequence[int],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular', 'none'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Apply the separable filter kernels to the tensor x along the axes axis.

    Does padding to keep the output the same size as the input.

    Parameters
    ----------
    x
        Tensor to filter
    kernels
        List of 1D kernels to apply to the tensor x
    axis
        Axes to filter over. Must have the same length as kernels.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant
    """
    if len(axis) != len(kernels):
        raise ValueError('Must provide matching length kernels and axis arguments.')

    # normalize axis to allow negative indexing in input
    axis = tuple([a % x.ndim for a in axis])
    if len(axis) != len(set(axis)):
        raise ValueError(f'Axis must be unique. Normalized axis are {axis}')

    # for pad_mode = constant and pad_value = 0, padding is done inside convolution, otherwise pad() is used.
    if pad_mode == 'constant' and pad_value == 0:
        padding_conv = 'same'
    else:
        padding_conv = 'valid'

    for kernel, ax in zip(kernels, axis, strict=False):
        # either both are complex or both are real
        if x.is_complex() and not kernel.is_complex():
            kernel = kernel + 0.0j
        elif kernel.is_complex() and not x.is_complex():
            x = x + 0.0j
        kernel = kernel.to(x.device)
        # moveaxis is not implemented for batched tensors, so vmap would fail.
        # thus we use permute.
        idx = list(range(x.ndim))
        # swapping the last axis and the axis to filter over
        idx[ax], idx[-1] = idx[-1], idx[ax]
        x = x.permute(idx)
        x_shape = list(x.shape)
        if pad_mode == 'none':
            x_shape[-1] -= len(kernel) - 1
        # flatten first to allow for circular, replicate and reflection padding for arbitrary tensor size
        x = x.flatten(end_dim=-2)
        if padding_conv == 'valid' and pad_mode != 'none':
            left_pad = (len(kernel) - 1) // 2
            right_pad = (len(kernel) - 1) - left_pad
            x = torch.nn.functional.pad(x, pad=(left_pad, right_pad), mode=pad_mode, value=pad_value)
        x = torch.nn.functional.conv1d(x[:, None, :], kernel[None, None, :], padding=padding_conv).reshape(x_shape)
        # for a single permutation, this undoes the permutation
        x = x.permute(idx)
    return x


def gaussian_filter(
    x: torch.Tensor,
    sigmas: float | Sequence[float] | torch.Tensor,
    axis: int | Sequence[int] | None = None,
    truncate: int = 3,
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Apply a and-Gaussian filter.

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
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    sigmas = torch.as_tensor(sigmas) if np.iterable(sigmas) else torch.tensor([sigmas] * len(axis))
    if not torch.all(sigmas > 0):
        raise ValueError('`sigmas` must be positive')

    if len(sigmas) != len(axis):
        raise ValueError('Must provide matching length sigmas and axis arguments. ')

    kernels = tuple(
        [
            torch.exp(-0.5 * (torch.arange(-ceil(truncate * sigma), ceil(truncate * sigma) + 1) / sigma) ** 2)
            for sigma in sigmas
        ]
    )
    kernels = tuple([(k / k.sum()).to(device=x.device) for k in kernels])
    x_filtered = filter_separable(x, kernels, axis, pad_mode, pad_value)
    return x_filtered


def uniform_filter(
    x: torch.Tensor,
    width: int | Sequence[int] | torch.Tensor,
    axis: int | Sequence[int] | None = None,
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Apply a and-uniform filter.

    Parameters
    ----------
    x
        Tensor to filter
    width
        Width of uniform kernel. If iterable, must have length equal to the number of axes.
    axis
        Axis or axes to filter over. If None, filters over all axes.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    width = torch.as_tensor(width) if np.iterable(width) else torch.tensor([width] * len(axis))
    if not torch.all(width > 0):
        raise ValueError('width must be positive.')
    if torch.any(width % 2 != 1):
        warnings.warn('width should be odd.', stacklevel=2)
    if len(width) != len(axis):
        raise ValueError('Must provide matching length width and axis arguments. ')
    width = torch.minimum(width, torch.tensor(x.shape)[(axis), ...])

    kernels = tuple([torch.ones(width, device=x.device) / width for width in width])
    x_filtered = filter_separable(x, kernels, axis, pad_mode, pad_value)
    return x_filtered
