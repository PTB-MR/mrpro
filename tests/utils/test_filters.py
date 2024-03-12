"""Tests for filters."""

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

import pytest
import torch
from mrpro.data import SpatialDimension
from mrpro.utils.filters import spatial_uniform_filter_3d


@pytest.fixture()
def data():
    """Create a simple 3D tensor with a single voxel set to 1.0."""
    data = torch.zeros(1, 1, 5, 5, 5)
    data[..., 2, 2, 2] = 1.0
    return data


def test_spatial_uniform_filter_3d(data):
    """Test spatial_uniform_filter_3d with SpatialDimension."""

    res = spatial_uniform_filter_3d(data, SpatialDimension(3, 3, 3))
    assert torch.sum(res) == torch.sum(data)


def test_spatial_uniform_filter_3d_tuple(data):
    """Test spatial_uniform_filter_3d with tuple."""

    res = spatial_uniform_filter_3d(data, (3, 3, 3))
    assert torch.sum(res) == torch.sum(data)


def test_spatial_uniform_filter_wrong_width(data):
    """Test spatial_uniform_filter_3d with wrong width."""

    with pytest.raises(ValueError, match='Invalid filter width'):
        spatial_uniform_filter_3d(data, (3, 3))
