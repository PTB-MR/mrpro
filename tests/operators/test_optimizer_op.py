import torch
from mrpro.operators import ConstraintsOp, FunctionalType, OptimizerOp
from mrpro.operators.functionals import L2NormSquared
from mrpro.operators.models import InversionRecovery
from mrpro.utils import RandomGenerator


def test_optimizer_op_gradcheck() -> None:
    """Test the optimizer op with gradcheck."""
    rng = RandomGenerator(seed=42)
    constraints_op = ConstraintsOp(
        bounds=(
            (-1, 1),  # M0 is not constrained
            (0.001, 4.0),  # T1 is constrained between 1 ms and 3 s
        )
    ).double()  # everything is double, otherwise the numerical derivative used in gradcheck gives wrong values

    rng = RandomGenerator(seed=1)
    true_m0 = rng.complex128_tensor(size=(3, 2))
    true_t1 = rng.float64_tensor(size=(3, 2), low=0.1, high=2)
    (signal,) = InversionRecovery(torch.tensor([0.5, 1.0, 1.5, 3], dtype=torch.float64))(true_m0, true_t1)
    t1_reg = true_t1 + rng.rand_like(true_t1, low=-0.01, high=0.01)
    m0_reg = true_m0 + rng.rand_like(true_m0, high=0.01)
    m0_reg.requires_grad = True

    def factory(
        m0_reg: torch.Tensor,
        t1_reg: torch.Tensor,
        lambda_m0: torch.Tensor,
        lambda_t1: torch.Tensor,
        signal: torch.Tensor,
    ) -> FunctionalType[torch.Tensor, torch.Tensor]:
        dc = L2NormSquared(signal) @ InversionRecovery((0.5, 1.0, 1.5, 3)).double()
        reg = lambda_m0 * L2NormSquared(m0_reg) | lambda_t1 * L2NormSquared(t1_reg)
        return (dc + reg) @ constraints_op

    op = OptimizerOp(
        factory=factory,
        initializer=lambda m0_reg, t1_reg, *_: constraints_op.inverse(m0_reg, t1_reg),
    )
    lambda_m0 = torch.tensor(1, requires_grad=True, dtype=torch.float64)
    lambda_t1 = torch.tensor(1, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradcheck(
        op, (m0_reg, t1_reg, lambda_m0, lambda_t1, signal), fast_mode=True, atol=1e-3, rtol=1e-2, eps=1e-3
    )

    ret = (constraints_op @ op)(m0_reg, t1_reg, lambda_m0, lambda_t1, signal)
    torch.testing.assert_close(ret[0], true_m0, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(ret[1], true_t1, atol=1e-3, rtol=1e-2)
