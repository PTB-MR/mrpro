import torch
from mrpro.operators import Jacobian
from mrpro.operators.functionals import L2NormSquared

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def test_jacobian_adjointness() -> None:
    """Test adjointness of Jacobian operator."""
    rng = RandomGenerator(123)
    x = rng.float32_tensor(3)
    y = rng.float32_tensor(())
    x0 = rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    dotproduct_adjointness_test(jacobian, x, y)


def test_jacobian_taylor():
    """Test Taylor expansion"""
    rng = RandomGenerator(123)
    x0 = rng.float32_tensor(3)
    x = x0 + 1e-2 * rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    fx = jacobian.taylor(x)
    torch.testing.assert_close(fx, op(x), rtol=1e-3, atol=1e-3)


def test_jacobian_gaussnewton():
    """Test Gauss Newton approximation of the Hessian"""
    rng = RandomGenerator(123)
    x0 = rng.float32_tensor(3)
    x = x0 + 1e-2 * rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    (actual,) = jacobian.gauss_newton(x)
    expected = torch.vdot(x, x0) * 4 * x0  # analytical solution for L2NormSquared
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_jacobian_value_at_x0():
    """Test value at x0"""
    rng = RandomGenerator(123)
    x0 = rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    (actual,) = jacobian.value_at_x0
    (expected,) = op(x0)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
