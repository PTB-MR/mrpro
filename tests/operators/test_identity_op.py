"""Tests for Identity Linear Operator and MultiIdentity Operator."""

import pytest
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


@pytest.mark.cuda
def test_identity_op_cuda():
    """Test identity operator works with CUDA devices."""

    # Generate input
    generator = RandomGenerator(seed=0)
    tensor = generator.complex64_tensor(2, 3, 4)

    # Create on CPU, run on CPU
    identity_op = IdentityOp()
    (y,) = identity_op(tensor)
    assert y.is_cpu

    # Transfer to GPU, run on GPU
    identity_op = IdentityOp()
    identity_op.cuda()
    (y,) = identity_op(tensor.cuda())
    assert y.is_cuda


@pytest.mark.cuda
def test_multi_identity_op_cuda():
    """Test multi identity operator works with CUDA devices."""

    # Generate input
    generator = RandomGenerator(seed=0)
    tensor1 = generator.complex64_tensor(2, 3, 4)
    tensor2 = generator.complex64_tensor(2, 3, 4)

    # Create on CPU, run on CPU
    multi_op = MultiIdentityOp()
    (y1, y2) = multi_op(tensor1, tensor2)
    assert y1.is_cpu
    assert y2.is_cpu

    # Transfer to GPU, run on GPU
    multi_op = MultiIdentityOp()
    multi_op.cuda()
    (y1, y2) = multi_op(tensor1.cuda(), tensor2.cuda())
    assert y1.is_cuda
    assert y2.is_cuda
