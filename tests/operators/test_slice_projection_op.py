"""Tests for projection operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from mrpro.data import SpatialDimension
from mrpro.operators import SliceProjectionOp


def test_slice_projection_op_basic():
    input_shape = SpatialDimension(10, 20, 30)
    slice_rotation = None
    slice_shift = 0.0
    slice_profile = 1.0
    operator = SliceProjectionOp(
        input_shape=input_shape, slice_rotation=slice_rotation, slice_shift=slice_shift, slice_profile=slice_profile
    )
    volume = torch.ones(input_shape.zyx)
    (slice2d,) = operator(volume)
    assert slice2d.shape == (1, 30, 30)
    expected = torch.zeros(1, 30, 30)
    expected[:, 5:-5, :] = 1
    torch.testing.assert_close(slice2d, expected)
