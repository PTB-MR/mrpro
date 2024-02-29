"""Tests for the operators module."""

import torch

from mrpro.operators import LinearOperator
from mrpro.operators import Operator


class DummyOperator(Operator):
    """Dummy operator for testing, raises input to the power of value."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self._value = value

    def forward(self, x: torch.Tensor):
        """Dummy operator."""
        return x**self._value


class DummyLinearOperator(LinearOperator):
    """Dummy linear operator for testing, multiplies input by value."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self._value = value

    def forward(self, x: torch.Tensor):
        """Dummy linear operator."""
        return x * 2

    def adjoint(self, x: torch.Tensor):
        """Dummy adjoint linear operator."""
        if x.is_complex():
            x = x.conj()
        if self._value.is_complex():
            return x * self._value.conj()
        return x * self._value


def test_composition_operator():
    a = DummyOperator(torch.tensor(2.0))
    b = DummyOperator(torch.tensor(3.0))
    c = a @ b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(b(x))

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator)
    assert not isinstance(c, LinearOperator)


def test_composition_linearoperator():
    a = DummyLinearOperator(torch.tensor(2.0))
    b = DummyLinearOperator(torch.tensor(3.0))
    c = a @ b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(b(x))

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator @ LinearOperator should be an Operator'
    assert isinstance(c, LinearOperator), 'LinearOperator @ LinearOperator should be a LinearOperator'


def test_composition_linearoperator_operator():
    a = DummyLinearOperator(torch.tensor(2.0))
    b = DummyOperator(torch.tensor(3.0))
    c = a @ b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(b(x))

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator @ Operator should be an Operator'
    assert not isinstance(c, LinearOperator), 'LinearOperator @ Operator should not be a LinearOperator'


def test_sum_operator():
    a = DummyOperator(torch.tensor(2.0))
    b = DummyOperator(torch.tensor(3.0))
    c = a + b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) + b(x)

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator + Operator should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator + Operator should not be a LinearOperator'


def test_sum_linearoperator():
    a = DummyLinearOperator(torch.tensor(2.0))
    b = DummyLinearOperator(torch.tensor(3.0))
    c = a + b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) + b(x)

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator + LinearOperator should be an Operator'
    assert isinstance(c, LinearOperator), 'LinearOperator + LinearOperator should be a LinearOperator'


def test_sum_linearoperator_operator():
    a = DummyLinearOperator(torch.tensor(2.0))
    b = DummyOperator(torch.tensor(3.0))
    c = a + b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) + b(x)

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator + Operator should be an Operator'
    assert not isinstance(c, LinearOperator), 'LinearOperator + Operator should not be a LinearOperator'


def test_sum_operator_linearoperator():
    a = DummyOperator(torch.tensor(3.0))
    b = DummyLinearOperator(torch.tensor(2.0))
    c = a + b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) + b(x)

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator + LinearOperator should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator + LinearOperator should not be a LinearOperator'


def test_elementwise_product_operator():
    a = DummyOperator(torch.tensor(2.0))
    b = torch.tensor(3.0)
    c = a * b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) * b

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator * scalar should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator * scalar should not be a LinearOperator'


def test_elementwise_rproduct_operator():
    a = DummyOperator(torch.tensor(2.0))
    b = torch.tensor(3.0)
    c = b * a
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) * b

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator * scalar should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator * scalar should not be a LinearOperator'


def test_elementwise_product_linearoperator():
    a = DummyLinearOperator(torch.tensor(2.0))
    b = torch.tensor(3.0)
    c = a * b
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) * b

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator * scalar should be an Operator'
    assert isinstance(c, LinearOperator), 'LinearOperator * scalar should be a LinearOperator'


def test_elementwise_rproduct_linearoperator():
    a = DummyLinearOperator(torch.tensor(2.0))
    b = torch.tensor(3.0)
    c = b * a
    x = torch.arange(10)
    y1 = c(x)
    y2 = a(x) * b

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'LinearOperator * scalar should be an Operator'
    assert isinstance(c, LinearOperator), 'LinearOperator * scalar should be a LinearOperator'
