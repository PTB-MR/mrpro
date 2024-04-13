"""Tests for the operators module."""

import pytest
import torch
from mrpro.operators import LinearOperator
from mrpro.operators import Operator

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


class DummyOperator(Operator[torch.Tensor, torch.Tensor]):
    """Dummy operator for testing, raises input to the power of value and sums."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self._value = value

    def forward(self, x: torch.Tensor):
        """Dummy operator."""
        return ((x**self._value).sum().unsqueeze(0),)


class DummyLinearOperator(LinearOperator):
    """Dummy linear operator for testing, multiplies input by value."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self._value = value

    def forward(self, x: torch.Tensor):
        """Dummy linear operator."""
        return (self._value @ x,)

    def adjoint(self, x: torch.Tensor):
        """Dummy adjoint linear operator."""
        return (self._value.mH @ x,)


def test_composition_operator():
    a = DummyOperator(torch.tensor(2.0))
    b = DummyOperator(torch.tensor(3.0))
    c = a @ b
    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    (y2,) = a(*b(x))

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator)
    assert not isinstance(c, LinearOperator)


def test_composition_linearoperator():
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((2, 3)))
    b = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    c = a @ b
    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    (y2,) = a(*b(x))

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator @ LinearOperator should be an Operator'
    assert isinstance(c, LinearOperator), 'LinearOperator @ LinearOperator should be a LinearOperator'


def test_composition_linearoperator_operator():
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((3, 1)))
    b = DummyOperator(torch.tensor(3.0))
    c = a @ b
    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    (y2,) = a(*b(x))

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator @ Operator should be an Operator'
    assert not isinstance(c, LinearOperator), 'LinearOperator @ Operator should not be a LinearOperator'


def test_sum_operator():
    a = DummyOperator(torch.tensor(2.0))
    b = DummyOperator(torch.tensor(3.0))
    c = a + b
    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] + b(x)[0]

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator + Operator should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator + Operator should not be a LinearOperator'


def test_sum_linearoperator():
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    c = a + b
    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] + b(x)[0]

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator + LinearOperator should be an Operator'
    assert isinstance(c, LinearOperator), 'LinearOperator + LinearOperator should be a LinearOperator'


def test_sum_linearoperator_operator():
    rng = RandomGenerator(0)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = DummyOperator(torch.tensor(3.0))
    c = a + b
    x = rng.complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] + b(x)[0]

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator + Operator should be an Operator'
    assert not isinstance(c, LinearOperator), 'LinearOperator + Operator should not be a LinearOperator'


def test_sum_operator_linearoperator():
    rng = RandomGenerator(0)
    a = DummyOperator(torch.tensor(3.0))
    b = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    c = a + b
    x = rng.complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] + b(x)[0]

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator + LinearOperator should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator + LinearOperator should not be a LinearOperator'


@pytest.mark.parametrize('value', [2, 3j])
def test_elementwise_product_operator(value):
    a = DummyOperator(torch.tensor(2.0))
    b = torch.ones(10) * value
    c = a * b
    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    (y2,) = a(b * x)

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator * tensor should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator * tensor should not be a LinearOperator'


@pytest.mark.parametrize('value', [2, 3j])
def test_elementwise_rproduct_operator(value):
    a = DummyOperator(torch.tensor(2.0))
    b = torch.tensor(value)
    c = b * a
    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    y2 = b * a(x)[0]

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'tensor * Operator should be an Operator'
    assert not isinstance(c, LinearOperator), 'tensor * Operator should not be a LinearOperator'


@pytest.mark.parametrize('value', [2, 3j])
def test_elementwise_product_linearoperator(value):
    rng = RandomGenerator(0)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = torch.ones(10) * value
    c = a * b
    assert isinstance(c, Operator), 'LinearOperator * tensor should be an Operator'
    assert isinstance(c, LinearOperator), 'LinearOperator * tensor should be a LinearOperator'

    x = rng.complex64_tensor(10)
    (y1,) = c(x)
    (y2,) = a(x * b)
    torch.testing.assert_close(y1, y2)

    y = rng.complex64_tensor(3)
    (x1,) = c.H(y)
    x2 = b.conj() * a.H(y)[0]
    torch.testing.assert_close(x1, x2)


@pytest.mark.parametrize('value', [2, 3j])
def test_elementwise_rproduct_linearoperator(value):
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = torch.ones(3) * value
    c = b * a
    assert isinstance(c, Operator), 'tensor * LinearOperator should be an Operator'
    assert isinstance(c, LinearOperator), 'tensor * LinearOperator should be a LinearOperator'

    x = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] * b
    torch.testing.assert_close(y1, y2)

    y = RandomGenerator(0).complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] * b
    torch.testing.assert_close(y1, y2)

    y = rng.complex64_tensor(3)
    (x1,) = c.H(y)
    (x2,) = a.H(b.conj() * y)
    torch.testing.assert_close(x1, x2)


def test_adjoint_composition_operators():
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((2, 3)))
    b = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    u = rng.complex64_tensor(10)
    v = rng.complex64_tensor(2)
    linear_op_composition = a @ b
    dotproduct_adjointness_test(linear_op_composition, u, v)


def test_adjoint_product_left():
    rng = RandomGenerator(0)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = rng.float32_tensor(10)
    u = rng.complex64_tensor(10)
    v = rng.complex64_tensor(3)
    linear_op_product = a * b
    dotproduct_adjointness_test(linear_op_product, u, v)


def test_adjoint_product_right():
    rng = RandomGenerator(1)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = rng.float32_tensor(3)
    u = rng.complex64_tensor(10)
    v = rng.complex64_tensor(3)
    linear_op_product = b * a
    dotproduct_adjointness_test(linear_op_product, u, v)


def test_adjoint_sum():
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    u = rng.complex64_tensor(10)
    v = rng.complex64_tensor(3)
    linear_op_sum = a + b
    dotproduct_adjointness_test(linear_op_sum, u, v)
