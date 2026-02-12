"""Tests for sliding windows."""

import pytest
import torch
from mr2.utils import RandomGenerator
from mr2.utils.sliding_window import sliding_window


def alternative_implementation(data, window_shape, axis, stride):
    """
    Alternative implementation of sliding window for testing purposes.
    No error handling; uses unfold for each axis.
    """
    result = data
    for ax, ws, st in zip(axis, window_shape, stride, strict=False):
        result = result.unfold(ax, ws, st)
    return result


@pytest.mark.parametrize(
    ('shape', 'window_shape', 'axis', 'stride'),
    [
        ((5, 6, 7), (3, 3, 3), (0, 1, 2), (1, 1, 1)),
        ((5, 6), (2, 2), (0, 1), (1, 1)),
        ((5, 6), (1, 1), (0, 1), (2, 3)),
    ],
)
def test_window_contents(shape, window_shape, axis, stride):
    """Test that the window contents are correct."""
    data = torch.arange(int(torch.tensor(shape).prod())).reshape(shape)
    window = sliding_window(data, window_shape, axis, stride)
    alt_window = alternative_implementation(data, window_shape, axis, stride)
    torch.testing.assert_close(window, alt_window)


def test_repeated_axes() -> None:
    """Test that repeated axes raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='Duplicate values'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, 0))
    with pytest.raises(ValueError, match='Duplicate values'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, -3))


def test_negative_window_shape() -> None:
    """Test that negative window shapes raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='window_shape must be positive'):
        _ = sliding_window(data, window_shape=(2, -2), dim=(0, 1))
    with pytest.raises(ValueError, match='window_shape must be positive'):
        _ = sliding_window(data, window_shape=-2, dim=(0, 1))


def test_negative_strides() -> None:
    """Test that negative strides raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='stride must be positive'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, 1), stride=(1, -1))
    with pytest.raises(ValueError, match='stride must be positive'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, 1), stride=-1)


def test_negative_dilation() -> None:
    """Test that negative dilation raises an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='dilation must be positive'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, 1), dilation=(1, -1))
    with pytest.raises(ValueError, match='dilation must be positive'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, 1), dilation=-1)


def test_length_mismatch() -> None:
    """Test that length mismatches raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='Length mismatch'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, 1, 2), stride=(1, 1))
    with pytest.raises(ValueError, match='Length mismatch'):
        _ = sliding_window(data, window_shape=(2, 2), dim=(0, 1), stride=(1, 1, 1))
    with pytest.raises(ValueError, match='Length mismatch'):
        _ = sliding_window(data, window_shape=2, dilation=(1, 1))


def test_window_too_large() -> None:
    """Test that too large windows raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='too small'):
        _ = sliding_window(data, window_shape=2, dim=-3, stride=1, dilation=2)


def test_scalar_arguments() -> None:
    """Test that scalar arguments are correctly interpreted."""
    data = RandomGenerator(42).float32_tensor((5, 6, 7))
    tuple_result = sliding_window(data, window_shape=(2,), dim=(1,), stride=(2,), dilation=(2,))
    int_result = sliding_window(data, window_shape=2, dim=1, stride=2, dilation=2)
    torch.testing.assert_close(tuple_result, int_result)


def test_isview() -> None:
    """Test that the window is a view of the original data."""
    data = RandomGenerator(42).float32_tensor((5, 6, 7))
    window = sliding_window(data, window_shape=(2, 2, 2), dim=(0, 1, 2), stride=(1, 1, 1))
    assert not window.is_contiguous()
    assert window.data_ptr() == data.data_ptr()
    expected_shape = (4, 5, 6, 2, 2, 2)
    assert window.shape == expected_shape


def test_dilation() -> None:
    """Test a dilated sliding window."""
    data = torch.arange(4)
    window = sliding_window(data, window_shape=2, dilation=2)
    expected = torch.tensor([[0, 2], [1, 3]])
    assert torch.equal(window, expected)
