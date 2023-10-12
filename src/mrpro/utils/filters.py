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
