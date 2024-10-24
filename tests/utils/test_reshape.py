"""Tests for reshaping utilities."""

import torch
from mrpro.utils import broadcast_right, reduce_view, unsqueeze_left, unsqueeze_right

from tests import RandomGenerator


def test_broadcast_right():
    """Test broadcast_right"""
    tensors = (torch.ones(1, 2, 3), torch.ones(1, 2), torch.ones(2))
    broadcasted = broadcast_right(*tensors)
    assert broadcasted[0].shape == broadcasted[1].shape == broadcasted[2].shape == (2, 2, 3)


def test_unsqueeze_left():
    """Test unsqueeze_left"""
    tensor = torch.ones(1, 2, 3)
    unsqueezed = unsqueeze_left(tensor, 2)
    assert unsqueezed.shape == (1, 1, 1, 2, 3)
    assert torch.equal(tensor.ravel(), unsqueezed.ravel())


def test_unsqueeze_right():
    """Test unsqueeze_right"""
    tensor = torch.ones(1, 2, 3)
    unsqueezed = unsqueeze_right(tensor, 2)
    assert unsqueezed.shape == (1, 2, 3, 1, 1)
    assert torch.equal(tensor.ravel(), unsqueezed.ravel())


def test_reduce_view():
    """Test reduce_view"""

    tensor = RandomGenerator(0).float32_tensor((1, 2, 3, 1, 1, 1))
    tensor = tensor.expand(1, 2, 3, 4, 1, 1).contiguous()  # this cannot be removed
    tensor = tensor.expand(7, 2, 3, 4, 5, 6)

    reduced_all = reduce_view(tensor)
    assert reduced_all.shape == (1, 2, 3, 4, 1, 1)
    assert torch.equal(reduced_all.expand_as(tensor), tensor)

    reduced_two = reduce_view(tensor, (0, -1))
    assert reduced_two.shape == (1, 2, 3, 4, 5, 1)
    assert torch.equal(reduced_two.expand_as(tensor), tensor)

    reduced_one_neg = reduce_view(tensor, -1)
    assert reduced_one_neg.shape == (7, 2, 3, 4, 5, 1)
    assert torch.equal(reduced_one_neg.expand_as(tensor), tensor)

    reduced_one_pos = reduce_view(tensor, 0)
    assert reduced_one_pos.shape == (1, 2, 3, 4, 5, 6)
    assert torch.equal(reduced_one_pos.expand_as(tensor), tensor)
