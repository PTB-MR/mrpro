"""Tests the Spatial class."""

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


@pytest.mark.cuda
def test_spatial_dimension_cuda_tensor():
    """Test moving to CUDA"""
    spatial_dimension = SpatialDimension(z=torch.ones(1), y=torch.ones(1), x=torch.ones(1))
    spatial_dimension_cuda = spatial_dimension.cuda()
    assert spatial_dimension_cuda.z.is_cuda
    assert spatial_dimension_cuda.y.is_cuda
    assert spatial_dimension_cuda.x.is_cuda
    assert spatial_dimension.z.is_cpu
    assert spatial_dimension.y.is_cpu
    assert spatial_dimension.x.is_cpu
    assert spatial_dimension_cuda.is_cuda
    assert spatial_dimension.is_cpu
    assert not spatial_dimension_cuda.is_cpu
    assert not spatial_dimension.is_cuda


def test_spatial_dimension_cuda_float():
    """Test moving to CUDA without tensors -> copy only"""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    # the device number should not matter, has there is no
    # data to move to the device
    spatial_dimension_cuda = spatial_dimension.cuda(42)
    # if a dataclass has no tensors, it is both on CPU and CUDA
    # and the device is None
    assert spatial_dimension_cuda.is_cuda
    assert spatial_dimension.is_cpu
    assert spatial_dimension_cuda.is_cpu
    assert spatial_dimension.is_cuda
    assert spatial_dimension.device is None
    assert spatial_dimension_cuda.device is None
    assert spatial_dimension_cuda is not spatial_dimension
