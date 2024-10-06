from typing import assert_type

import torch
from mrpro.operators import IdentityOp, LinearOperator, MagnitudeOp, Operator, ZeroOp

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def test_zero_op():
    """Test that the zero operator returns zeros."""
    generator = RandomGenerator(seed=0)
    x = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp()
    (actual,) = operator(x)
    expected = torch.zeros_like(x)
    torch.testing.assert_close(actual, expected)


def test_zero_op_neutral_linop():
    """Test that the zero operator is neutral for addition."""
    op = IdentityOp()
    zero = ZeroOp()

    rsum = op + zero
    lsum = zero + op
    assert rsum is op
    assert lsum is op
    assert_type(lsum, LinearOperator)
    assert_type(rsum, LinearOperator)

    assert zero + zero is zero

    assert 1 * zero is zero
    assert zero * 1 is zero


def test_zero_op_neutral_op():
    """Test that the zero operator is neutral for addition."""
    op = MagnitudeOp() @ IdentityOp()
    zero = ZeroOp()
    rsum = op + zero
    lsum = zero + op
    assert rsum is op
    assert lsum is op
    assert_type(rsum, Operator[torch.Tensor, tuple[torch.Tensor]])
    assert_type(lsum, Operator[torch.Tensor, tuple[torch.Tensor]])


def test_zero_op_adjoint():
    """Test that the adjoint of the zero operator is the zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp()
    dotproduct_adjointness_test(operator, u, v)
