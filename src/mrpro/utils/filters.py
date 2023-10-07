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

import numpy as np
import torch

from mrpro.data import SpatialDimension


def spatial_uniform_filter_3d(data: torch.Tensor, filter_width: SpatialDimension[int]) -> torch.Tensor:
    """Spatial smoothing using convolution with box function.

    Parameters
    ----------
    data
        Data to be smoothed in the shape (... z y x)
    filter_width
        Width of 3D the filter
    """
    # Ensure (batch z y x)
    ddim = data.shape
    data = data.view(-1, *ddim[-3:])

    # Create a box-shaped filter kernel
    filter_width_zyx = [filter_width.z, filter_width.y, filter_width.x]
    kernel = torch.ones(filter_width_zyx) / np.prod(filter_width_zyx)
    kernel = kernel.to(dtype=data.dtype)
    kernel = kernel.view(1, 1, *kernel.size())

    # Use same filter kernel for each channel
    kernel = kernel.repeat(data.shape[0], *[1] * (kernel.dim() - 1))
    return torch.reshape(torch.nn.functional.conv3d(data, weight=kernel, groups=data.shape[0], padding='same'), ddim)
