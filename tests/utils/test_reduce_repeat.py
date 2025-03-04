"""Test reduce_repeat"""

import pytest
import torch
from mrpro.utils import reduce_repeat

from tests import RandomGenerator


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
def test_reduce_repeat_expanded(dtype: torch.dtype) -> None:
    """Test reduction of broadcasted dimensions."""
    rng = RandomGenerator(13)
    original = rng.rand_tensor((2, 1, 1, 5), dtype=dtype, low=0.0, high=1.0)
    expanded = original.expand(2, 2, 2, 5)
    result = reduce_repeat(expanded, tol=0)
    assert result.shape == (2, 1, 1, 5)
    assert (result == original).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('tol', [0.001, 0.1])
def test_reduce_repeat_tol(dtype: torch.dtype, tol: float) -> None:
    """Test reduction of noisy dimensions"""
    rng = RandomGenerator(14)
    original = rng.rand_tensor((5, 1, 1, 5), dtype=dtype, low=0.0, high=1.0)
    expanded = original.expand(5, 2, 2, 5)
    expanded = expanded + rng.rand_tensor(expanded.shape, dtype=expanded.dtype, low=0, high=0.5 * tol)
    result = reduce_repeat(expanded, tol=tol)
    assert result.shape == (5, 1, 1, 5)
    assert (result == expanded[:, :1, :1, :]).all()


def test_reduce_repeat_dim() -> None:
    """Test dimension selection"""
    rng = RandomGenerator(15)
    original = rng.float32_tensor((1, 1, 1, 1))
    expanded = original.expand(5, 2, 2, 5)
    result = reduce_repeat(expanded, tol=0, dim=(1, -2))
    assert result.shape == (5, 1, 1, 5)
