import torch
from mrpro.operators import Jacobian
from mrpro.operators.functionals import L2NormSquared

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def test_jacobian_adjointness():
    rng = RandomGenerator(123)
    x = rng.float32_tensor(3)
    y = rng.float32_tensor(())
    x0 = rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    dotproduct_adjointness_test(jacobian, x, y)


def test_jacobian_taylor():
    rng = RandomGenerator(123)
    x0 = rng.float32_tensor(3)
    x = x0 + 1e-2 * rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    fx = jacobian.taylor(x)
    torch.testing.assert_allclose(fx, op(x), rtol=1e-3, atol=1e-3)
