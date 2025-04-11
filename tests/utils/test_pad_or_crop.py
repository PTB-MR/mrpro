"""Tests for padding and cropping of data tensors."""

import torch
from mrpro.utils import RandomGenerator
from mrpro.utils.pad_or_crop import pad_or_crop


def test_pad_or_crop_content():
    """Test changing data by cropping and padding."""
    generator = RandomGenerator(seed=0)
    original_data_shape = (100, 200, 50)
    new_data_shape = (80, 100, 240)
    original_data = generator.complex64_tensor(original_data_shape)
    new_data = pad_or_crop(original_data, new_data_shape, dim=(-3, -2, -1), value=123)

    # Compare overlapping region
    torch.testing.assert_close(original_data[10:90, 50:150, :], new_data[:, :, 95:145])
    assert new_data[0, 0, 0] == 123
