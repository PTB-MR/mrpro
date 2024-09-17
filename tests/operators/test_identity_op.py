"""Tests for Identity Linear Operator."""

import torch
from mrpro.operators import IdentityOp

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
