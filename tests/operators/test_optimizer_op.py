from collections.abc import Callable

import torch
from mrpro.operators import OptimizerOp
from mrpro.operators.functionals import L2NormSquared
from mrpro.operators.models import InversionRecovery
from mrpro.utils import RandomGenerator


def test_optimizer_op():
    rng = RandomGenerator(seed=0)

    def factory(
        m0_reg: torch.Tensor,
        t1_reg: torch.Tensor,
        lambda_m0: torch.Tensor,
        lambda_t1: torch.Tensor,
        signal: torch.Tensor,
    ) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        data_consistency = L2NormSquared(signal) @ InversionRecovery((0.5, 1.0, 1.5, 3))
        regularization = lambda_t1 * L2NormSquared(t1_reg) | lambda_m0 * L2NormSquared(m0_reg)
        return data_consistency + regularization

    factory(torch.randn(10, 10), torch.randn(10, 10), torch.randn(1), torch.randn(1), torch.randn(10, 10))
    rng = RandomGenerator(seed=0)
    true_m0 = rng.complex64_tensor(size=(10, 10))
    true_t1 = rng.float32_tensor(size=(10, 10), low=0.1, high=2)
    (signal,) = InversionRecovery((0.5, 1.0, 1.5, 3))(true_m0, true_t1)
    signal += rng.complex64_tensor(size=(10, 10), high=0.1)
    t1_reg = true_t1 + rng.rand_like(true_t1, low=-0.1, high=0.1)
    m0_reg = true_m0 + rng.rand_like(true_m0, low=0, high=0.5)
    op = OptimizerOp(factory, initializer=lambda m0_reg, t1_reg, *_: (m0_reg, t1_reg))
    lambda_m0 = torch.tensor(1.0, requires_grad=True)
    lambda_t1 = torch.tensor(1.0, requires_grad=True)
    ret = op.forward(m0_reg, t1_reg, lambda_m0, lambda_t1, signal)
    (loss,) = (L2NormSquared(true_m0) | L2NormSquared(m0_reg))(*ret)
    loss.backward()
    assert lambda_m0.grad is not None
    assert lambda_t1.grad is not None
