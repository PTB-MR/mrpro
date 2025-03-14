"""Tests the Spatial class."""

import pytest
import torch
from mrpro.data import SpatialDimension
from typing_extensions import Any, assert_type

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
        x = 1 * torch.ones(1, 2, 3)
        y = 2 * torch.ones(1, 2, 3)
        z = 3 * torch.ones(1, 2, 3)

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


def test_from_array_arraylike():
    """Test creation from an ArrayLike list of list of int"""
    xyz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    spatial_dimension_xyz = SpatialDimension.from_array_xyz(list(zip(*xyz, strict=False)))
    assert isinstance(spatial_dimension_xyz.x, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.y, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.z, torch.Tensor)
    assert_type(spatial_dimension_xyz, SpatialDimension[torch.Tensor])
    assert torch.equal(spatial_dimension_xyz.x, torch.tensor(xyz[0]))
    assert torch.equal(spatial_dimension_xyz.y, torch.tensor(xyz[1]))
    assert torch.equal(spatial_dimension_xyz.z, torch.tensor(xyz[2]))

    spatial_dimension_zyx = SpatialDimension.from_array_zyx(list(zip(*xyz[::-1], strict=False)))
    assert isinstance(spatial_dimension_xyz.x, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.y, torch.Tensor)
    assert isinstance(spatial_dimension_xyz.z, torch.Tensor)
    assert_type(spatial_dimension_zyx, SpatialDimension[torch.Tensor])
    assert torch.equal(spatial_dimension_zyx.x, torch.tensor(xyz[0]))
    assert torch.equal(spatial_dimension_zyx.y, torch.tensor(xyz[1]))
    assert torch.equal(spatial_dimension_zyx.z, torch.tensor(xyz[2]))


def test_spatial_dimension_from_array_wrongshape():
    """Test error message on wrong shape"""
    tensor_wrongshape = torch.zeros(1, 2, 5)
    with pytest.raises(ValueError, match='last dimension'):
        _ = SpatialDimension.from_array_xyz(tensor_wrongshape)


def test_spatial_dimension_broadcasting():
    z = torch.ones(2, 1, 1)
    y = torch.ones(1, 2, 1)
    x = torch.ones(1, 1, 2)
    spatial_dimension = SpatialDimension(z, y, x)
    assert spatial_dimension.shape == (2, 2, 2)


def test_spatial_dimension_apply_():
    """Test apply_ (in place)"""

    def conversion(x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), 'The argument to the conversion function should be a tensor'
        return x.swapaxes(0, 1).square()

    xyz = RandomGenerator(0).float32_tensor((1, 2, 3))
    spatial_dimension = SpatialDimension.from_array_xyz(xyz.numpy())
    spatial_dimension_inplace = spatial_dimension.apply_().apply_(conversion)

    assert spatial_dimension_inplace is spatial_dimension

    assert isinstance(spatial_dimension_inplace.x, torch.Tensor)
    assert isinstance(spatial_dimension_inplace.y, torch.Tensor)
    assert isinstance(spatial_dimension_inplace.z, torch.Tensor)

    x, y, z = conversion(xyz).unbind(-1)
    assert torch.equal(spatial_dimension_inplace.x, x)
    assert torch.equal(spatial_dimension_inplace.y, y)
    assert torch.equal(spatial_dimension_inplace.z, z)


def test_spatial_dimension_apply():
    """Test apply (out of place)"""

    def conversion(x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), 'The argument to the conversion function should be a tensor'
        return x.swapaxes(0, 1).square()

    xyz = RandomGenerator(0).float32_tensor((1, 2, 3))
    spatial_dimension = SpatialDimension.from_array_xyz(xyz.numpy())
    spatial_dimension_outofplace = spatial_dimension.apply(conversion)

    assert spatial_dimension_outofplace is not spatial_dimension

    assert isinstance(spatial_dimension_outofplace.x, torch.Tensor)
    assert isinstance(spatial_dimension_outofplace.y, torch.Tensor)
    assert isinstance(spatial_dimension_outofplace.z, torch.Tensor)

    x, y, z = conversion(xyz).unbind(-1)
    assert torch.equal(spatial_dimension_outofplace.x, x)
    assert torch.equal(spatial_dimension_outofplace.y, y)
    assert torch.equal(spatial_dimension_outofplace.z, z)

    x, y, z = xyz.unbind(-1)  # original should be unmodified
    assert torch.equal(spatial_dimension.x, x)
    assert torch.equal(spatial_dimension.y, y)
    assert torch.equal(spatial_dimension.z, z)


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


@pytest.mark.cuda
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


def test_spatial_dimension_getitem_tensor():
    """Test accessing elements of SpatialDimension."""
    zyx = RandomGenerator(0).float32_tensor((4, 2, 3))
    spatial_dimension = SpatialDimension.from_array_zyx(zyx)
    torch.testing.assert_close(torch.stack(spatial_dimension[:2, ...].zyx, dim=-1), zyx[:2, ...])


def test_spatial_dimension_setitem_tensor():
    """Test setting elements of SpatialDimension[torch.Tensor]."""
    zyx = RandomGenerator(0).float32_tensor((4, 2, 3))
    spatial_dimension = SpatialDimension.from_array_zyx(zyx)
    spatial_dimension_to_set = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension[2, 1] = spatial_dimension_to_set
    assert spatial_dimension[2, 1].zyx == spatial_dimension_to_set.zyx


def test_spatial_dimension_mul():
    """Test multiplication of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_mul = spatial_dimension * value
    assert isinstance(spatial_dimension_mul, SpatialDimension)
    assert spatial_dimension_mul.zyx == (
        spatial_dimension.z * value,
        spatial_dimension.y * value,
        spatial_dimension.x * value,
    )


def test_spatial_dimension_rmul():
    """Test right multiplication of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_mul = value * spatial_dimension
    assert isinstance(spatial_dimension_mul, SpatialDimension)
    assert spatial_dimension_mul.zyx == (
        spatial_dimension.z * value,
        spatial_dimension.y * value,
        spatial_dimension.x * value,
    )


def test_spatial_dimension_mul_spatial_dimension():
    """Test multiplication of SpatialDimension with SpatialDimension."""
    spatial_dimension_1 = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_2 = SpatialDimension(z=4.0, y=5.0, x=6.0)
    spatial_dimension_mul = spatial_dimension_1 * spatial_dimension_2
    assert isinstance(spatial_dimension_mul, SpatialDimension)
    assert spatial_dimension_mul.zyx == (
        spatial_dimension_1.z * spatial_dimension_2.z,
        spatial_dimension_1.y * spatial_dimension_2.y,
        spatial_dimension_1.x * spatial_dimension_2.x,
    )


def test_spatial_dimension_truediv():
    """Test division of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_div = spatial_dimension / value
    assert isinstance(spatial_dimension_div, SpatialDimension)
    assert spatial_dimension_div.zyx == (
        spatial_dimension.z / value,
        spatial_dimension.y / value,
        spatial_dimension.x / value,
    )


def test_spatial_dimension_rtruediv():
    """Test right division of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_div = value / spatial_dimension
    assert isinstance(spatial_dimension_div, SpatialDimension)
    assert spatial_dimension_div.zyx == (
        value / spatial_dimension.z,
        value / spatial_dimension.y,
        value / spatial_dimension.x,
    )


def test_spatial_dimension_truediv_spatial_dimension():
    """Test divitions of SpatialDimension with SpatialDimension."""
    spatial_dimension_1 = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_2 = SpatialDimension(z=4.0, y=5.0, x=6.0)
    spatial_dimension_div = spatial_dimension_1 / spatial_dimension_2
    assert isinstance(spatial_dimension_div, SpatialDimension)
    assert spatial_dimension_div.zyx == (
        spatial_dimension_1.z / spatial_dimension_2.z,
        spatial_dimension_1.y / spatial_dimension_2.y,
        spatial_dimension_1.x / spatial_dimension_2.x,
    )


def test_spatial_dimension_add():
    """Test addition of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_add = spatial_dimension + value
    assert isinstance(spatial_dimension_add, SpatialDimension)
    assert spatial_dimension_add.zyx == (
        spatial_dimension.z + value,
        spatial_dimension.y + value,
        spatial_dimension.x + value,
    )


def test_spatial_dimension_radd():
    """Test right addition of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_add = value + spatial_dimension
    assert isinstance(spatial_dimension_add, SpatialDimension)
    assert spatial_dimension_add.zyx == (
        spatial_dimension.z + value,
        spatial_dimension.y + value,
        spatial_dimension.x + value,
    )


def test_spatial_dimension_add_spatial_dimension():
    """Test addition of SpatialDimension with SpatialDimension."""
    spatial_dimension_1 = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_2 = SpatialDimension(z=4.0, y=5.0, x=6.0)
    spatial_dimension_add = spatial_dimension_1 + spatial_dimension_2
    assert isinstance(spatial_dimension_add, SpatialDimension)
    assert spatial_dimension_add.zyx == (
        spatial_dimension_1.z + spatial_dimension_2.z,
        spatial_dimension_1.y + spatial_dimension_2.y,
        spatial_dimension_1.x + spatial_dimension_2.x,
    )


def test_spatial_dimension_sub():
    """Test subtraction of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_sub = spatial_dimension - value
    assert isinstance(spatial_dimension_sub, SpatialDimension)
    assert spatial_dimension_sub.zyx == (
        spatial_dimension.z - value,
        spatial_dimension.y - value,
        spatial_dimension.x - value,
    )


def test_spatial_dimension_rsub():
    """Test right subtraction of SpatialDimension with numeric value."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    value = 3
    spatial_dimension_sub = value - spatial_dimension
    assert isinstance(spatial_dimension_sub, SpatialDimension)
    assert spatial_dimension_sub.zyx == (
        value - spatial_dimension.z,
        value - spatial_dimension.y,
        value - spatial_dimension.x,
    )


def test_spatial_dimension_sub_spatial_dimension():
    """Test subtraction of SpatialDimension with SpatialDimension."""
    spatial_dimension_1 = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_2 = SpatialDimension(z=4.0, y=5.0, x=6.0)
    spatial_dimension_sub = spatial_dimension_1 - spatial_dimension_2
    assert isinstance(spatial_dimension_sub, SpatialDimension)
    assert spatial_dimension_sub.zyx == (
        spatial_dimension_1.z - spatial_dimension_2.z,
        spatial_dimension_1.y - spatial_dimension_2.y,
        spatial_dimension_1.x - spatial_dimension_2.x,
    )


def test_spatial_dimension_eq_float():
    """Test equality of SpatialDimension."""
    eq = SpatialDimension(z=1.0, y=2.0, x=3.0) == SpatialDimension(z=1, y=2, x=3)
    assert_type(eq, bool)
    assert eq
    neq = SpatialDimension(z=1.0, y=2.0, x=3.0) == SpatialDimension(z=1.0, y=2.0, x=4.0)
    assert_type(neq, bool)
    assert not neq


def test_spatial_dimension_eq_tensor():
    """Test equality of SpatialDimension with tensors."""
    spatial_dimension_1 = SpatialDimension(z=torch.ones(2), y=torch.ones(2), x=torch.ones(2))
    spatial_dimension_2 = SpatialDimension(z=torch.ones(2), y=torch.ones(2), x=torch.arange(2))
    comp: Any = spatial_dimension_1 == spatial_dimension_2
    assert torch.equal(comp, torch.tensor([False, True]))


def test_spatial_dimension_comp_scalar():
    """Test equality of SpatialDimension."""
    assert SpatialDimension(1, 2, 3) > SpatialDimension(0, 0, 0)
    assert SpatialDimension(1, 2, 3) >= SpatialDimension(1, 0, 0)
    assert not SpatialDimension(1, 2, 3) > SpatialDimension(1, 0, 0)

    assert SpatialDimension(1, 2, 3) < SpatialDimension(10, 10, 10)
    assert SpatialDimension(1, 2, 3) <= SpatialDimension(10, 10, 3)
    assert not SpatialDimension(1, 2, 3) < SpatialDimension(10, 10, 3)


def test_spatial_dimension_comp_tensor():
    """Test equality of SpatialDimension."""
    t = torch.ones(2)
    assert (SpatialDimension(1 * t, 2 * t, 3 * t) > SpatialDimension(0 * t, 0 * t, 0 * t)).all()
    assert (SpatialDimension(1 * t, 2 * t, 3 * t) >= SpatialDimension(1 * t, 0 * t, 0 * t)).all()
    assert not (SpatialDimension(1 * t, 2 * t, 3 * t) > SpatialDimension(1 * t, 0 * t, 0 * t)).any()

    assert (SpatialDimension(1 * t, 2 * t, 3 * t) < SpatialDimension(10 * t, 10 * t, 10 * t)).all()
    assert (SpatialDimension(1 * t, 2 * t, 3 * t) <= SpatialDimension(10 * t, 10 * t, 3 * t)).all()
    assert not (SpatialDimension(1 * t, 2 * t, 3 * t) < SpatialDimension(10 * t, 10 * t, 3 * t)).any()


def test_spatial_dimension_neg():
    """Test negation of SpatialDimension."""
    spatial_dimension = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_neg = -spatial_dimension
    assert isinstance(spatial_dimension_neg, SpatialDimension)
    assert spatial_dimension_neg.zyx == (-spatial_dimension.z, -spatial_dimension.y, -spatial_dimension.x)


def mypy_test_spatial_dimension_typing_add():
    """Test typing of SpatialDimension operations (mypy)

    This test checks that the typing of the operations is correct.
    It will be used by pytest, but mypy will complain if any of the
    types are wrong.
    """
    spatial_dimension_float = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_int = SpatialDimension(z=1, y=2, x=3)

    spatial_dimension_tensor = SpatialDimension(z=torch.ones(1), y=torch.ones(1), x=torch.ones(1))
    scalar_float = 1.0
    scalar_int = 1
    scalar_tensor = torch.ones(1)

    # int
    assert_type(spatial_dimension_int + spatial_dimension_int, SpatialDimension[int])
    assert_type(spatial_dimension_int + scalar_int, SpatialDimension[int])
    assert_type(scalar_int + spatial_dimension_int, SpatialDimension[int])

    # tensor
    assert_type(spatial_dimension_tensor + spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor + scalar_tensor, SpatialDimension[torch.Tensor])
    # assert_type(scalar_tensor + spatial_dimension_tensor, SpatialDimension[torch.Tensor]) # FIXME torch typing issue # noqa

    # float
    assert_type(spatial_dimension_float + spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float + scalar_float, SpatialDimension[float])
    assert_type(scalar_float + spatial_dimension_float, SpatialDimension[float])

    # int gets promoted to float
    assert_type(spatial_dimension_int + spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float + spatial_dimension_int, SpatialDimension[float])
    assert_type(spatial_dimension_int + scalar_float, SpatialDimension[float])
    assert_type(scalar_float + spatial_dimension_int, SpatialDimension[float])

    # int or float gets promoted to tensor
    assert_type(spatial_dimension_int + spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor + spatial_dimension_int, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_float + spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor + spatial_dimension_float, SpatialDimension[torch.Tensor])

    assert_type(scalar_int + spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor + scalar_int, SpatialDimension[torch.Tensor])
    assert_type(scalar_float + spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor + scalar_float, SpatialDimension[torch.Tensor])


def mypy_test_spatial_dimension_typing_sub():
    """Test typing of SpatialDimension operations (mypy)

    This test checks that the typing of the operations is correct.
    It will be used by pytest, but mypy will complain if any of the
    types are wrong.
    """
    spatial_dimension_float = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_int = SpatialDimension(z=1, y=2, x=3)

    spatial_dimension_tensor = SpatialDimension(z=torch.ones(1), y=torch.ones(1), x=torch.ones(1))
    scalar_float = 1.0
    scalar_int = 1
    scalar_tensor = torch.ones(1)

    # int
    assert_type(spatial_dimension_int - spatial_dimension_int, SpatialDimension[int])
    assert_type(spatial_dimension_int - scalar_int, SpatialDimension[int])
    assert_type(scalar_int - spatial_dimension_int, SpatialDimension[int])

    # tensor
    assert_type(spatial_dimension_tensor - spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor - scalar_tensor, SpatialDimension[torch.Tensor])
    # assert_type(scalar_tensor - spatial_dimension_tensor, SpatialDimension[torch.Tensor]) # FIXME torch typing issue # noqa

    # float
    assert_type(spatial_dimension_float - spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float - scalar_float, SpatialDimension[float])
    assert_type(scalar_float - spatial_dimension_float, SpatialDimension[float])

    # int gets promoted to float
    assert_type(spatial_dimension_int - spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float - spatial_dimension_int, SpatialDimension[float])
    assert_type(spatial_dimension_int - scalar_float, SpatialDimension[float])
    assert_type(scalar_float - spatial_dimension_int, SpatialDimension[float])

    # int or float gets promoted to tensor
    assert_type(spatial_dimension_int - spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor - spatial_dimension_int, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_float - spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor - spatial_dimension_float, SpatialDimension[torch.Tensor])

    assert_type(scalar_int - spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor - scalar_int, SpatialDimension[torch.Tensor])
    assert_type(scalar_float - spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor - scalar_float, SpatialDimension[torch.Tensor])


def mypy_test_spatial_dimension_typing_floordiv():
    """Test typing of SpatialDimension operations (mypy)

    This test checks that the typing of the operations is correct.
    It will be used by pytest, but mypy will complain if any of the
    types are wrong.
    """
    spatial_dimension_float = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_int = SpatialDimension(z=1, y=2, x=3)

    spatial_dimension_tensor = SpatialDimension(z=torch.ones(1), y=torch.ones(1), x=torch.ones(1))
    scalar_float = 1.0
    scalar_int = 1
    scalar_tensor = torch.ones(1)

    # int
    assert_type(spatial_dimension_int // spatial_dimension_int, SpatialDimension[int])
    assert_type(spatial_dimension_int // scalar_int, SpatialDimension[int])
    assert_type(scalar_int // spatial_dimension_int, SpatialDimension[int])

    # tensor
    assert_type(spatial_dimension_tensor // spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor // scalar_tensor, SpatialDimension[torch.Tensor])
    # assert_type(scalar_tensor // spatial_dimension_tensor, SpatialDimension[torch.Tensor]) # FIXME torch typing issue # noqa

    # float
    assert_type(spatial_dimension_float // spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float // scalar_float, SpatialDimension[float])
    assert_type(scalar_float // spatial_dimension_float, SpatialDimension[float])

    # int gets promoted to float
    assert_type(spatial_dimension_int // spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float // spatial_dimension_int, SpatialDimension[float])
    assert_type(spatial_dimension_int // scalar_float, SpatialDimension[float])
    assert_type(scalar_float // spatial_dimension_int, SpatialDimension[float])

    # int or float gets promoted to tensor
    assert_type(spatial_dimension_int // spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor // spatial_dimension_int, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_float // spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor // spatial_dimension_float, SpatialDimension[torch.Tensor])

    assert_type(scalar_int // spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor // scalar_int, SpatialDimension[torch.Tensor])
    assert_type(scalar_float // spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor // scalar_float, SpatialDimension[torch.Tensor])


def mypy_test_spatial_dimension_typing_mul():
    """Test typing of SpatialDimension operations (mypy)

    This test checks that the typing of the operations is correct.
    It will be used by pytest, but mypy will complain if any of the
    types are wrong.
    """
    spatial_dimension_float = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_int = SpatialDimension(z=1, y=2, x=3)

    spatial_dimension_tensor = SpatialDimension(z=torch.ones(1), y=torch.ones(1), x=torch.ones(1))
    scalar_float = 1.0
    scalar_int = 1
    scalar_tensor = torch.ones(1)

    # int
    assert_type(spatial_dimension_int * spatial_dimension_int, SpatialDimension[int])
    assert_type(spatial_dimension_int * scalar_int, SpatialDimension[int])
    assert_type(scalar_int * spatial_dimension_int, SpatialDimension[int])

    # tensor
    assert_type(spatial_dimension_tensor * spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor * scalar_tensor, SpatialDimension[torch.Tensor])
    # assert_type(scalar_tensor * spatial_dimension_tensor, SpatialDimension[torch.Tensor]) # FIXME torch typing issue # noqa

    # float
    assert_type(spatial_dimension_float * spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float * scalar_float, SpatialDimension[float])
    assert_type(scalar_float * spatial_dimension_float, SpatialDimension[float])

    # int gets promoted to float
    assert_type(spatial_dimension_int * spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float * spatial_dimension_int, SpatialDimension[float])
    assert_type(spatial_dimension_int * scalar_float, SpatialDimension[float])
    assert_type(scalar_float * spatial_dimension_int, SpatialDimension[float])

    # int or float gets promoted to tensor
    assert_type(spatial_dimension_int * spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor * spatial_dimension_int, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_float * spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor * spatial_dimension_float, SpatialDimension[torch.Tensor])

    assert_type(scalar_int * spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor * scalar_int, SpatialDimension[torch.Tensor])
    assert_type(scalar_float * spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor * scalar_float, SpatialDimension[torch.Tensor])


def mypy_test_spatial_dimension_typing_truediv():
    """Test typing of SpatialDimension operations (mypy)

    This test checks that the typing of the operations is correct.
    It will be used by pytest, but mypy will complain if any of the
    types are wrong.
    """
    spatial_dimension_float = SpatialDimension(z=1.0, y=2.0, x=3.0)
    spatial_dimension_int = SpatialDimension(z=1, y=2, x=3)

    spatial_dimension_tensor = SpatialDimension(z=torch.ones(1), y=torch.ones(1), x=torch.ones(1))
    scalar_float = 1.0
    scalar_int = 1
    scalar_tensor = torch.ones(1)

    # tensor
    assert_type(spatial_dimension_tensor / spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor / scalar_tensor, SpatialDimension[torch.Tensor])
    # assert_type(scalar_tensor / spatial_dimension_tensor, SpatialDimension[torch.Tensor]) # FIXME torch typing issue # noqa

    # float
    assert_type(spatial_dimension_float / spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float / scalar_float, SpatialDimension[float])
    assert_type(scalar_float / spatial_dimension_float, SpatialDimension[float])

    # int gets promoted to float
    assert_type(spatial_dimension_int / spatial_dimension_int, SpatialDimension[float])
    assert_type(spatial_dimension_int / scalar_int, SpatialDimension[float])
    assert_type(scalar_int / spatial_dimension_int, SpatialDimension[float])
    assert_type(spatial_dimension_int / spatial_dimension_float, SpatialDimension[float])
    assert_type(spatial_dimension_float / spatial_dimension_int, SpatialDimension[float])
    assert_type(spatial_dimension_int / scalar_float, SpatialDimension[float])
    assert_type(scalar_float / spatial_dimension_int, SpatialDimension[float])

    # int or float gets promoted to tensor
    assert_type(spatial_dimension_int / spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor / spatial_dimension_int, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_float / spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor / spatial_dimension_float, SpatialDimension[torch.Tensor])

    assert_type(scalar_int / spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor / scalar_int, SpatialDimension[torch.Tensor])
    assert_type(scalar_float / spatial_dimension_tensor, SpatialDimension[torch.Tensor])
    assert_type(spatial_dimension_tensor / scalar_float, SpatialDimension[torch.Tensor])


def test_spatial_dimension_masked_inplace_add():
    """Test inplace add and masking."""
    spatial_dimension = SpatialDimension(z=torch.arange(3), y=torch.arange(3), x=torch.arange(3))
    mask = spatial_dimension < SpatialDimension(z=torch.tensor(1), y=torch.tensor(1), x=torch.tensor(1))
    spatial_dimension[mask] += SpatialDimension(z=1, y=2, x=3)
    assert torch.equal(spatial_dimension.z, torch.tensor([1, 1, 2]))
    assert torch.equal(spatial_dimension.y, torch.tensor([2, 1, 2]))
    assert torch.equal(spatial_dimension.x, torch.tensor([3, 1, 2]))
