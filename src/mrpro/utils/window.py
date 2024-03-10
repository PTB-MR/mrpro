"""Sliding window view."""

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


def sliding_window(
    x,
    window_shape,
    axis=None,
    strides=1,
) -> torch.Tensor:
    """Sliding window view into the tensor x.

    Returns a view into the tensor x that represents a sliding window.

    Parameters
    ----------
    x
        Tensor to slide over
    window_shape
        Size of window over each axis that takes part in the sliding window.
    axis
        Axis or axes to slide over. If None, slides over all axes.
    strides
        Stride of the sliding window. **Experimental**.
    """

    if axis is None:
        axis = tuple(range(x.ndim))
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,) * len(axis)
    strides = tuple(strides) if np.iterable(strides) else (strides,) * len(axis)
    window_shape_arr = torch.tensor(window_shape)
    strides_arr = torch.tensor(strides)
    x_shape_arr = torch.tensor(x.shape)

    if torch.any(strides_arr != 1):
        warnings.warn('strides other than 1 are not fully supported')
    if torch.any(window_shape_arr < 0):
        raise ValueError('window_shape cannot contain negative values')
    if torch.any(strides_arr < 0):
        raise ValueError('strides cannot contain negative values')
    if len(window_shape) != len(axis):
        raise ValueError('Must provide matching length window_shape and axis arguments. ')
    if len(strides) != len(axis):
        raise ValueError('Must provide matching length strides and axis arguments.')

    out_strides = torch.tensor([x.stride(i) for i in range(x.ndim)] + [x.stride(ax) for ax in axis])
    out_strides[axis,] = out_strides[axis,] * strides_arr
    x_shape_arr[axis,] = (x_shape_arr[axis,] + strides_arr - 1) // strides_arr
    if torch.any(x_shape_arr < 0):
        raise ValueError('strides or windows too large')
    out_shape = tuple(x_shape_arr) + window_shape
    view = x.as_strided(size=out_shape, stride=tuple(out_strides))
    return view
