"""Tests for sliding windows."""

import pytest
import torch
from mrpro.utils.sliding_window import sliding_window

from tests import RandomGenerator


def alternative_implementation(data, window_shape, axis, stride):
    """Alternative implementation of sliding window for testing purposes.

    no error handling, expanded parameters, no fancy indexing, copies
    """
    for ax, ws, st in zip(axis, window_shape, stride, strict=False):
        data = data.unfold(ax, ws, st)
    return data


@pytest.mark.filterwarnings('ignore:strides other than 1 are not fully supported')
@pytest.mark.parametrize(
    ('shape', 'window_shape', 'axis', 'stride'),
    [
        ((5, 6, 7), (3, 3, 3), (0, 1, 2), (1, 1, 1)),
        ((5, 6), (2, 2), (1, 0), (1, 1)),
        ((5, 6), (1, 1), (0, 1), (2, 3)),
    ],
)
def test_window_contents(shape, window_shape, axis, stride):
    """Test that the contents of the window are correct."""
    # we use this pattern for easier debugging than random inputs
    data = torch.arange(int(torch.tensor(shape).prod())).reshape(shape)
    window = sliding_window(data, window_shape, axis, stride)
    alt_window = alternative_implementation(data, window_shape, axis, stride)
    torch.testing.assert_close(window, alt_window)


def test_repeated_axes():
    """Test that repeated axes raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='duplicate value'):
        _ = sliding_window(data, window_shape=(2, 2), axis=(0, 0))
    with pytest.raises(ValueError, match='duplicate value'):
        _ = sliding_window(data, window_shape=(2, 2), axis=(0, -3))


def test_negative_window_shape():
    """Test that negative window shapes raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='negative values'):
        _ = sliding_window(data, window_shape=(2, -2), axis=(0, 1))
    with pytest.raises(ValueError, match='negative values'):
        _ = sliding_window(data, window_shape=-2, axis=(0, 1))


@pytest.mark.filterwarnings('ignore:strides other than 1 are not fully supported')
def test_negative_strides():
    """Test that negative strides raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='negative values'):
        _ = sliding_window(data, window_shape=(2, 2), axis=(0, 1), strides=(1, -1))
    with pytest.raises(ValueError, match='negative values'):
        _ = sliding_window(data, window_shape=(2, 2), axis=(0, 1), strides=-1)


def test_length_mismatch():
    """Test that length mismatches raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='matching length'):
        _ = sliding_window(data, window_shape=(2, 2), axis=(0, 1, 2), strides=(1, 1))
    with pytest.raises(ValueError, match='matching length'):
        _ = sliding_window(data, window_shape=(2, 2), axis=(0, 1), strides=(1, 1, 1))


def test_window_too_large():
    """Test that too large windows raise an error."""
    data = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match='too large'):
        _ = sliding_window(data, window_shape=3, axis=0, strides=1)


@pytest.mark.filterwarnings('ignore:strides other than 1 are not fully supported')
def test_scalar_arguments():
    """Test that scalar arguments are correctly interpreted."""
    data = RandomGenerator(42).float32_tensor((5, 6, 7))
    tuple_result = sliding_window(data, window_shape=(2,), axis=(1,), strides=(2,))
    int_result = sliding_window(data, window_shape=2, axis=1, strides=2)
    torch.testing.assert_close(tuple_result, int_result)


def test_isview():
    """Test that the window is a view of the original data."""
    data = RandomGenerator(42).float32_tensor((5, 6, 7))
    window = sliding_window(data, window_shape=(2, 2, 2), axis=(0, 1, 2), strides=(1, 1, 1))
    assert not window.is_contiguous()
    assert window.data_ptr() == data.data_ptr()
    assert window.shape == (4, 5, 6, 2, 2, 2)
