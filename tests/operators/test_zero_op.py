import torch
from mrpro.operators import IdentityOp, MagnitudeOp, ZeroOp

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

    assert op + zero is op
    assert zero + op is op

    assert zero + zero is zero

    assert 1 * zero is zero
    assert zero * 1 is zero


def test_zero_op_neutral_op():
    """Test that the zero operator is neutral for addition."""
    op = MagnitudeOp()
    zero = ZeroOp()

    assert op + zero is op
    assert zero + op is op


def test_zero_op_adjoint():
    """Test that the adjoint of the zero operator is the zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp()
    dotproduct_adjointness_test(operator, u, v)
