"""Tests for reshaping utilities."""

import torch
from mrpro.utils import broadcast_right, unsqueeze_left, unsqueeze_right


def test_broadcast_right():
    """Test broadcast_right"""
    tensors = (torch.ones(1, 2, 3), torch.ones(1, 2), torch.ones(2))
    broadcasted = broadcast_right(*tensors)
    assert broadcasted[0].shape == broadcasted[1].shape == broadcasted[2].shape == (2, 2, 3)


def test_unsqueeze_left():
    """Test unsqueeze left"""
    tensor = torch.ones(1, 2, 3)
    unsqueezed = unsqueeze_left(tensor, 2)
    assert unsqueezed.shape == (1, 1, 1, 2, 3)
    assert torch.equal(tensor.ravel(), unsqueezed.ravel())


def test_unsqueeze_right():
    """Test unsqueeze right"""
    tensor = torch.ones(1, 2, 3)
    unsqueezed = unsqueeze_right(tensor, 2)
    assert unsqueezed.shape == (1, 2, 3, 1, 1)
    assert torch.equal(tensor.ravel(), unsqueezed.ravel())
