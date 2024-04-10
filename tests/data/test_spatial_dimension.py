"""Tests the Spatial class."""

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

import pytest
import torch
from mrpro.data import SpatialDimension

from tests import RandomGenerator


def test_spatial_dimension_from_xyz_int():
    """Test creation from an object with x, y, z attributes"""

    class XYZint:
        x = 1
        y = 2
        z = 3

    spatial_dimension = SpatialDimension.from_xyz(XYZint)
    assert spatial_dimension.x == 1
    assert spatial_dimension.y == 2
    assert spatial_dimension.z == 3


def test_spatial_dimension_from_xyz_tensor():
    """Test creation from an object with x, y, z attributes"""

    class XYZtensor:
        x = 1 * torch.ones(1)
        y = 2 * torch.ones(2)
        z = 3 * torch.ones(3)

    spatial_dimension = SpatialDimension.from_xyz(XYZtensor())
    assert torch.equal(spatial_dimension.x, XYZtensor.x)
    assert torch.equal(spatial_dimension.y, XYZtensor.y)
    assert torch.equal(spatial_dimension.z, XYZtensor.z)


def test_spatial_dimension_from_array():
    """Test creation from arrays"""
    xyz = RandomGenerator(0).float32_tensor((1, 2, 3))
    spatial_dimension_xyz = SpatialDimension.from_array_xyz(xyz.numpy())
    assert isinstance(spatial_dimension_xyz.x, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.y, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.z, torch.Tensor)
    assert torch.equal(spatial_dimension_xyz.x, xyz[..., 0])
    assert torch.equal(spatial_dimension_xyz.y, xyz[..., 1])
    assert torch.equal(spatial_dimension_xyz.z, xyz[..., 2])

    zyx = torch.flip(xyz, dims=(-1,))
    spatial_dimension_zyx = SpatialDimension.from_array_zyx(zyx)
    assert torch.equal(spatial_dimension_xyz.x, spatial_dimension_zyx.x)
    assert torch.equal(spatial_dimension_xyz.y, spatial_dimension_zyx.y)
    assert torch.equal(spatial_dimension_xyz.z, spatial_dimension_zyx.z)


def test_spatial_dimension_from_array_wrongshape():
    """Test error message on wrong shape"""
    tensor_wrongshape = torch.zeros(1, 2, 5)
    with pytest.raises(ValueError, match='last dimension'):
        _ = SpatialDimension.from_array_xyz(tensor_wrongshape)


def test_spatial_dimension_from_array_conversion():
    """Test conversion argument"""

    def conversion(x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), 'The argument to the conversion function should be a tensor'
        return x.swapaxes(0, 1).square()

    xyz = RandomGenerator(0).float32_tensor((1, 2, 3))
    spatial_dimension_xyz = SpatialDimension.from_array_xyz(xyz.numpy(), conversion=conversion)
    assert isinstance(spatial_dimension_xyz.x, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.y, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.z, torch.Tensor)

    x, y, z = conversion(xyz).unbind(-1)
    assert torch.equal(spatial_dimension_xyz.x, x)
    assert torch.equal(spatial_dimension_xyz.y, y)
    assert torch.equal(spatial_dimension_xyz.z, z)


def test_spatial_dimension_zyx():
    """Test the zyx tuple property"""
    z, y, x = (2, 3, 4)
    spatial_dimension = SpatialDimension(z=z, y=y, x=x)
    assert isinstance(spatial_dimension.zyx, tuple)
    assert spatial_dimension.zyx == (z, y, x)
