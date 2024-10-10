"""Tests for filters."""

import pytest
import torch
from einops import repeat
from mrpro.utils.filters import filter_separable, gaussian_filter, uniform_filter


@pytest.fixture
def data():
    """Create a simple 3D tensor with a single voxel set to 1.0."""
    data = torch.zeros(1, 1, 5, 5, 5)
    data[..., 2, 2, 2] = 1.0
    return data


@pytest.mark.parametrize(
    ('pad_mode', 'center_value', 'edge_value'),
    [('constant', 44, 7), ('reflect', 44, 6), ('replicate', 44, 5), ('circular', 44, 24), ('none', 48, 8)],
)
def test_filter_separable(pad_mode, center_value, edge_value):
    """Test filter_separable and different padding modes."""

    data = repeat(torch.arange(1, 21), 'x -> y x', y=1).to(dtype=torch.float32)
    kernels = (torch.as_tensor([1.0, 2.0, 1.0]),)
    result = filter_separable(
        data, kernels, dim=(1,), pad_mode=pad_mode, pad_value=3.0 if pad_mode == 'constant' else 0.0
    )
    if pad_mode == 'none':
        assert result.shape == (data.shape[0], data.shape[1] - len(kernels[0]) + 1)
    else:
        assert data.shape == result.shape
    assert result[0, 10] == center_value
    assert result[0, 0] == edge_value


@pytest.mark.parametrize('filter_dtype', [torch.float32, torch.float64, torch.int32, torch.complex64, torch.complex128])
@pytest.mark.parametrize('data_dtype', [torch.float32, torch.float64, torch.int32, torch.complex64, torch.complex128])
def test_filter_separable_dtype(filter_dtype, data_dtype):
    """Test filter_separable and different padding modes."""

    data = repeat(torch.arange(1, 21), 'x -> y x', y=1).to(dtype=data_dtype)
    kernels = (torch.tensor([1, 2, 1], dtype=filter_dtype),)
    result = filter_separable(data, kernels, dim=(1,))
    expected_dtype = torch.result_type(data, kernels[0])
    assert result.dtype == expected_dtype
    assert result[0, 10] == 44


def test_gaussian_filter_int_axis(data):
    """Test Gaussian filter."""
    result = gaussian_filter(data, 0.5, -1)
    expected = torch.tensor(
        [
            [0, 0, 2, 2, 0],
            [0, 0, 2, 2, 1],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 2, 3],
            [0, 0, 2, 2, 4],
        ],
    )
    assert torch.equal(result.nonzero(), expected)
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_filter_two_axis(data):
    """Test Gaussian filter along two axes."""
    result = gaussian_filter(data, 0.5, (-1, 3))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_filter_two_sigmas(data):
    """Test Gaussian filter with different sigma values."""
    result = gaussian_filter(data, (0.2, 0.5), (-1, 3))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_filter_noaxis(data):
    """Test Gaussian filter is applied along all axes if not specified."""
    result = gaussian_filter(data, sigmas=torch.tensor(0.2))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_gaussian_invalid_sigmas(data):
    """Test Gaussian filter with invalid sigma values."""
    with pytest.raises(ValueError, match='positive'):
        gaussian_filter(data, dim=(-1, 2), sigmas=torch.tensor([0.2, 0.0]))
    with pytest.raises(ValueError, match='positive'):
        gaussian_filter(data, sigmas=torch.tensor(-1.0))
    with pytest.raises(ValueError, match='positive'):
        gaussian_filter(data, sigmas=torch.nan)
    with pytest.raises(ValueError, match='length'):
        gaussian_filter(data, sigmas=(1.0, 2.0))


def test_uniform_filter_int_axis(data):
    """Test uniform filter."""
    result = uniform_filter(data, 3, -1)
    assert result.shape == data.shape
    assert (result > 0).sum() == 3
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_uniform_filter_two_axis(data):
    """Test uniform filter along two axes."""
    result = uniform_filter(data, 3, (-1, -2))
    assert result.shape == data.shape
    assert (result > 0).sum() == 9
    assert (result[..., 2] > 0).sum() == 3
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_uniform_filter_two_widths(data):
    """Test uniform filter with two widths."""
    result = uniform_filter(data, (3, 5), (-1, -2))
    assert result.shape == data.shape
    assert (result > 0).sum() == 15
    assert (result[..., 2] > 0).sum() == 5
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_uniform_filter_noaxis(data):
    """Test uniform filter is applied along all axes if not specified."""
    result = uniform_filter(data, width=torch.tensor(3))
    assert result.shape == data.shape
    torch.testing.assert_close(result.sum(), torch.tensor(1.0), atol=1e-3, rtol=0)
    # result should be the same along different axes of same size
    assert torch.equal(result[0, 0, :, 2, 2], result[0, 0, 2, :, 2])
    assert torch.equal(result[0, 0, :, 2, 2], result[0, 0, 2, 2, :])


def test_uniform_invalid_width(data):
    """Test uniform filter with invalid width."""
    with pytest.raises(ValueError, match='positive'):
        uniform_filter(data, dim=(-1, 2), width=torch.tensor([3, 0]))
    with pytest.raises(ValueError, match='positive'):
        uniform_filter(data, width=torch.tensor(-1.0))
    with pytest.raises(ValueError, match='positive'):
        uniform_filter(data, width=torch.nan)  # type: ignore[arg-type]
    with pytest.warns(UserWarning, match='odd'):
        uniform_filter(data, width=2)
    with pytest.raises(ValueError, match='length'):
        uniform_filter(data, width=(3.0, 3.0))  # type: ignore[arg-type]
