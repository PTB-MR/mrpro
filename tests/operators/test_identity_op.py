"""Tests for Identity Linear Operator and MultiIdentity Operator."""

import torch
from mrpro.operators import IdentityOp, MagnitudeOp, MultiIdentityOp
from mrpro.operators.LinearOperator import LinearOperator
from typing_extensions import assert_type

from tests import RandomGenerator


def test_identity_op():
    """Test forward identity."""
    generator = RandomGenerator(seed=0)
    tensor = generator.complex64_tensor(2, 3, 4)
    operator = IdentityOp()
    torch.testing.assert_close(tensor, *operator(tensor))
    assert tensor is operator(tensor)[0]


def test_identity_op_adjoint():
    """Test adjoint identity."""
    generator = RandomGenerator(seed=0)
    tensor = generator.complex64_tensor(2, 3, 4)
    operator = IdentityOp().H
    torch.testing.assert_close(tensor, *operator(tensor))
    assert tensor is operator(tensor)[0]


def test_identity_op_operatorsyntax():
    """Test Identity@(Identity*alpha) + (beta*Identity.H).H"""
    generator = RandomGenerator(seed=0)
    tensor = generator.complex64_tensor(2, 3, 4)
    alpha = generator.complex64_tensor(2, 3, 4)
    beta = generator.complex64_tensor(2, 3, 4)
    composition = IdentityOp() @ (IdentityOp() * alpha) + (beta * IdentityOp().H).H
    expected = tensor * alpha + tensor * beta.conj()
    (actual,) = composition(tensor)
    torch.testing.assert_close(actual, expected)


def test_multi_identity_op():
    """Test forward multi identity."""
    generator = RandomGenerator(seed=0)
    tensor = generator.complex64_tensor(2, 3, 4)
    operator = MultiIdentityOp()
    torch.testing.assert_close(tuple(tensor), operator(*tensor))


def test_identity_is_neutral():
    class DummyLinearOperator(LinearOperator):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
            return (x,)

        def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
            return (x,)

    op = DummyLinearOperator()
    identity = IdentityOp()
    assert op @ identity is op
    assert identity @ op is op


def test_multi_identity_is_neutral():
    """Test that MultiIdentityOp is neutral for operator composition."""
    op = MagnitudeOp()
    identity = MultiIdentityOp()
    assert op @ identity is op
    assert identity @ op is op
    assert_type((op @ identity)(torch.ones(1), torch.ones(1)), tuple[torch.Tensor, torch.Tensor])
    assert_type((identity @ op)(torch.ones(1), torch.ones(1)), tuple[torch.Tensor, torch.Tensor])
