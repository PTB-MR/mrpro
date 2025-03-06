"""Tests for reshaping utilities."""

import numpy as np
import pytest
import torch
from mrpro.utils import (
    broadcast_right,
    ravel_multi_index,
    reduce_view,
    reshape_broadcasted,
    unsqueeze_at,
    unsqueeze_left,
    unsqueeze_right,
    unsqueeze_tensors_at,
    unsqueeze_tensors_left,
    unsqueeze_tensors_right,
)

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


def test_unsqueeze_tensors_left() -> None:
    """Test unsqueeze_tensors_left"""
    tensor1 = torch.ones(1, 2, 3)
    tensor2 = torch.ones(1, 2)
    tensor3 = torch.ones(3)
    unsqueezed = unsqueeze_tensors_left(tensor1, tensor2, tensor3)
    assert unsqueezed[0].shape == (1, 2, 3)
    assert unsqueezed[1].shape == (1, 1, 2)
    assert unsqueezed[2].shape == (1, 1, 3)


def test_unsqueeze_tensors_right() -> None:
    """Test unsqueeze_tensors_right"""
    tensor1 = torch.ones(1, 2, 3)
    tensor2 = torch.ones(1, 2)
    tensor3 = torch.ones(3)
    unsqueezed = unsqueeze_tensors_right(tensor1, tensor2, tensor3)
    assert unsqueezed[0].shape == (1, 2, 3)
    assert unsqueezed[1].shape == (1, 2, 1)
    assert unsqueezed[2].shape == (3, 1, 1)


def test_unsqueeze_tensors_left_ndim() -> None:
    """Test unsqueeze_tensors_left with ndim set"""
    tensor1 = torch.ones(1, 2, 3)
    tensor2 = torch.ones(1, 2)
    tensor3 = torch.ones(3)
    unsqueezed = unsqueeze_tensors_left(tensor1, tensor2, tensor3, ndim=4)
    assert unsqueezed[0].shape == (1, 1, 2, 3)
    assert unsqueezed[1].shape == (1, 1, 1, 2)
    assert unsqueezed[2].shape == (1, 1, 1, 3)


def test_unsqueeze_tensors_right_ndim() -> None:
    """Test unsqueeze_tensors_right with ndim set"""
    tensor1 = torch.ones(1, 2, 3)
    tensor2 = torch.ones(1, 2)
    tensor3 = torch.ones(3)
    unsqueezed = unsqueeze_tensors_right(tensor1, tensor2, tensor3, ndim=4)
    assert unsqueezed[0].shape == (1, 2, 3, 1)
    assert unsqueezed[1].shape == (1, 2, 1, 1)
    assert unsqueezed[2].shape == (3, 1, 1, 1)


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


@pytest.mark.parametrize(
    ('shape', 'expand_shape', 'permute', 'final_shape', 'expected_stride'),
    [
        ((1, 2, 3, 1, 1), (1, 2, 3, 4, 5), (0, 2, 1, 3, 4), (1, 6, 2, 2, 5), (6, 1, 0, 0, 0)),
        ((1, 2, 1), (100, 2, 2), (0, 1, 2), (100, 4), (0, 1)),
        ((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 0, 1), (1, 2, 6, 10, 1), (0, 0, 0, 0, 0)),
        ((1, 2, 3), (1, -1, 3), (0, 1, 2), (6,), (1,)),
    ],
)
def test_reshape_broadcasted(shape, expand_shape, permute, final_shape, expected_stride):
    """Test reshape_broadcasted"""
    rng = RandomGenerator(0)
    tensor = rng.float32_tensor(shape).expand(*expand_shape).permute(*permute)
    reshaped = reshape_broadcasted(tensor, *final_shape)
    expected_values = tensor.reshape(*final_shape)
    assert reshaped.shape == expected_values.shape
    assert reshaped.stride() == expected_stride
    assert torch.equal(reshaped, expected_values)


def test_reshape_broadcasted_fail():
    """Test reshape_broadcasted with invalid input"""
    a = torch.ones(2)
    with pytest.raises(RuntimeError, match='invalid'):
        reshape_broadcasted(a, 3)
    with pytest.raises(RuntimeError, match='invalid'):
        reshape_broadcasted(a, -1, -3)
    with pytest.raises(RuntimeError, match='only one dimension'):
        reshape_broadcasted(a, -1, -1)


def test_ravel_multidex() -> None:
    """Test ravel_multiindex"""
    rng = RandomGenerator(1)
    dims = [5, 1, 6]
    indices = [
        rng.int64_tensor((2, 3), low=0, high=dims[0]),
        rng.int64_tensor((1, 1), low=0, high=dims[1]),
        rng.int64_tensor((2, 1), low=0, high=dims[2]),
    ]
    expected = torch.as_tensor(np.ravel_multi_index([idx.numpy() for idx in indices], dims))
    actual = ravel_multi_index(indices, dims)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    ('shape', 'n', 'dim', 'expected'),
    [
        ((2, 3, 4), 0, 1, (2, 3, 4)),
        ((2, 3, 4), 1, 0, (1, 2, 3, 4)),
        ((2, 3, 4), 1, -4, (1, 2, 3, 4)),
        ((2, 3, 4), 1, -1, (2, 3, 4, 1)),
        ((2, 3, 4), 1, 3, (2, 3, 4, 1)),
        ((2, 3, 4), 1, -2, (2, 3, 1, 4)),
        ((2, 3, 4), 1, 1, (2, 1, 3, 4)),
        ((2, 3, 4), 2, 0, (1, 1, 2, 3, 4)),
        ((2, 3, 4), 2, -4, (1, 1, 2, 3, 4)),
        ((2, 3, 4), 2, -1, (2, 3, 4, 1, 1)),
        ((2, 3, 4), 2, 3, (2, 3, 4, 1, 1)),
        ((2, 3, 4), 2, -2, (2, 3, 1, 1, 4)),
        ((2, 3, 4), 2, 1, (2, 1, 1, 3, 4)),
    ],
)
def test_unsqueeze_at(shape: tuple[int, ...], n: int, dim: int, expected: tuple[int, ...]) -> None:
    """Test unsqueeze_at"""
    tensor = RandomGenerator(0).float32_tensor(shape)
    reshaped = unsqueeze_at(tensor, dim, n)
    assert reshaped.shape == expected
    assert torch.equal(tensor, reshaped.squeeze())


def test_unqueeze_at_errors() -> None:
    """Test unsqueeze_at with invalid input"""
    tensor = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        unsqueeze_at(tensor, dim=1, n=-1)
    with pytest.raises(IndexError):
        # unsqueeze shortcut due to n=1
        unsqueeze_at(tensor, dim=4, n=1)
    with pytest.raises(IndexError):
        # unsqueeze shortcut due to n=1
        unsqueeze_at(tensor, dim=-5, n=1)
    with pytest.raises(IndexError):
        # reshape due to n=2
        unsqueeze_at(tensor, dim=4, n=2)
    with pytest.raises(IndexError):
        # reshape due to n=2
        unsqueeze_at(tensor, dim=-5, n=2)


def test_unsqueeze_tensors_at() -> None:
    """Test unsqueeze_tensors_at"""
    rng = RandomGenerator(13)
    tensor1 = rng.float32_tensor((4, 2, 3))
    tensor2 = rng.float32_tensor((5, 2))
    tensor3 = rng.float32_tensor((3,))
    unsqueezed = unsqueeze_tensors_at(tensor1, tensor2, tensor3, dim=1)
    assert unsqueezed[0].shape == (4, 2, 3)
    assert unsqueezed[1].shape == (5, 1, 2)
    assert unsqueezed[2].shape == (3, 1, 1)
    assert torch.equal(unsqueezed[0].squeeze(), tensor1)
    assert torch.equal(unsqueezed[1].squeeze(), tensor2)
    assert torch.equal(unsqueezed[2].squeeze(), tensor3)


def test_unsqueeze_tensors_ndim() -> None:
    """Test unsqueeze_tensors_at with ndim set"""
    rng = RandomGenerator(13)
    tensor1 = rng.float32_tensor((4, 2, 3))
    tensor2 = rng.float32_tensor((5, 2))
    unsqueezed = unsqueeze_tensors_at(tensor1, tensor2, ndim=5, dim=2)
    assert unsqueezed[0].shape == (4, 2, 1, 1, 3)
    assert unsqueezed[1].shape == (5, 2, 1, 1, 1)
    assert torch.equal(unsqueezed[0].squeeze(), tensor1)
    assert torch.equal(unsqueezed[1].squeeze(), tensor2)
