"""Tests for Rearrange Operator."""

from mrpro.operators.PatchOp import PatchOp

from tests import RandomGenerator, dotproduct_adjointness_test


def test_patch_op_adjointness() -> None:
    """Test adjointness and shape of Rearrange Op."""
    rng = RandomGenerator(seed=0)
    input_shape = (3, 4, 5)
    output_shape = (2, 1, 3, 5)
    u = rng.complex64_tensor(size=input_shape)
    v = rng.complex64_tensor(size=output_shape)
    operator = PatchOp((0, 1), (1, 3), (3, 1), (2, 1), domain_size=input_shape[:2])
    (v,) = operator(u)
    dotproduct_adjointness_test(operator, u, v)
