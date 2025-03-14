"""Test the averaging operator."""

import pytest
import torch
from mrpro.operators import AveragingOp

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def test_averageingop_adjointness() -> None:
    """Test the adjointness of the averaging operator."""
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=(5, 15, 10))
    v = rng.complex64_tensor(size=(5, 3, 10))
    idx = [0, 1, 2], slice(0, 6, 2), torch.tensor([-1, -2, -3])
    op = AveragingOp(dim=-2, idx=idx, domain_size=15)
    dotproduct_adjointness_test(op, u, v)


def test_averageingop_forward() -> None:
    """Test the forward method of the averaging operator."""
    rng = RandomGenerator(seed=1)
    u = rng.complex64_tensor(size=(5, 10))
    idx = [0, 1, 2], slice(0, 6, 2), torch.tensor([-1, -2, -3])
    op = AveragingOp(dim=0, idx=idx, domain_size=5)
    (actual,) = op(u)
    torch.testing.assert_close(actual[0], u[(0, 1, 2),].mean(dim=0))
    torch.testing.assert_close(actual[1], u[(0, 2, 4),].mean(dim=0))
    torch.testing.assert_close(actual[2], u[(-1, -2, -3),].mean(dim=0))


def test_averageingop_no_domain_size() -> None:
    """Test the adjoint method of the averaging operator without domain size."""
    rng = RandomGenerator(seed=0)
    v = rng.float32_tensor(size=(5, 2))
    idx = rng.int64_tensor(size=(2, 3), low=0, high=9)
    op = AveragingOp(dim=1, idx=idx)

    with pytest.raises(ValueError, match='Domain'):
        op.adjoint(v)

    u = rng.float32_tensor(size=(5, 10))
    op(u)  # this sets the domain size
    with pytest.warns(match='Domain'):
        (actual,) = op.adjoint(v)
    assert actual.shape == u.shape
