"""Tests interpolation."""

import pytest
import torch
from mr2.utils.interpolate import apply_lowres, interpolate


@pytest.fixture
def data() -> torch.Tensor:
    """Create a simple 5D tensor with a linear ramp of step size 1."""
    data = torch.arange(0, 20).repeat(10, 10, 1).unsqueeze(0).unsqueeze(0)
    return data


@pytest.mark.parametrize('size', [10, 20, 30])
@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_interpolate_linear(data: torch.Tensor, size: int, data_dtype: torch.dtype) -> None:
    """Linear ramp should remain a linear ramp after inear interpolation."""
    result = interpolate(data.to(dtype=data_dtype), size=(size,), dim=(4,), mode='linear')
    assert result.dtype == data_dtype
    assert torch.diff(result[..., 1:-1], dim=-1).isclose(torch.tensor(data.shape[-1] / size, dtype=data_dtype)).all()


@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_interpolate_nearest(data: torch.Tensor, data_dtype: torch.dtype) -> None:
    """Tensor is unchanged after nearest upsampling and then downsampling."""
    result = interpolate(data.to(dtype=data_dtype), size=(data.shape[-1] * 2,), dim=(4,), mode='nearest')
    result = interpolate(result, size=(data.shape[-1],), dim=(4,), mode='nearest')
    torch.testing.assert_close(result, data.to(dtype=data_dtype))


def test_interpolate_size_dim_mismatch() -> None:
    """Test mismatch between size and dim."""
    with pytest.raises(ValueError, match='matching length'):
        interpolate(torch.randn((2, 2, 2)), dim=(-1, -2), size=(2,))


def test_interpolate_unique_dim() -> None:
    """Test non-unique interpolate dimensions."""
    with pytest.raises(IndexError, match='unique'):
        interpolate(torch.randn((2, 2, 2)), dim=(-1, -2, 1), size=(2, 2, 2))


def test_apply_lowres(data: torch.Tensor) -> None:
    """Applying identity function should not change linear ramp data except for edges."""
    data = data.to(torch.complex64)

    def identity_function(x: torch.Tensor):
        return x

    apply = apply_lowres(identity_function, size=(8, 8, 8), dim=(-3, -2, -1))
    torch.testing.assert_close(data[..., 1:-1], apply(data)[..., 1:-1])
