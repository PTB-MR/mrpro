"""Tests for padding and cropping of data tensors."""

from typing import Literal

import pytest
import torch
from mr2.utils import RandomGenerator
from mr2.utils.pad_or_crop import pad_or_crop


@pytest.mark.parametrize('mode', ['constant', 'reflect', 'replicate', 'circular'])
def test_pad_or_crop_content(mode: Literal['constant', 'reflect', 'replicate', 'circular']):
    """Test changing data by cropping and padding."""
    generator = RandomGenerator(seed=0)
    original_data_shape = (100, 200, 50)
    new_data_shape = (80, 100, 70)
    original_data = generator.complex64_tensor(original_data_shape)
    new_data = pad_or_crop(
        original_data, new_data_shape, dim=(-3, -2, -1), value=123 if mode == 'constant' else 0, mode=mode
    )

    # Compare overlapping region
    torch.testing.assert_close(original_data[10:90, 50:150, :], new_data[:, :, 10:60])
    # ...  and padded region
    match mode:
        case 'constant':
            assert new_data[0, 0, 0] == 123
        case 'reflect':
            assert new_data[0, 0, 9] == original_data[10, 50, 1]
        case 'replicate':
            assert new_data[0, 0, 9] == original_data[10, 50, 0]
        case 'circular':
            assert new_data[0, 0, 9] == original_data[10, 50, -1]
