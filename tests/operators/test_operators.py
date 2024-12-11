"""Tests for the operators module."""

from typing import cast

import pytest
import torch
from mrpro.operators import LinearOperator, Operator
from typing_extensions import Any, assert_type

from tests import RandomGenerator, dotproduct_adjointness_test


class DummyOperator(Operator[torch.Tensor, tuple[torch.Tensor,]]):
    """Dummy operator for testing, raises input to the power of value and sums."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self._value = value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
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
    assert_type(c, LinearOperator)


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
    assert_type(c, Operator[torch.Tensor, tuple[torch.Tensor]])


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
    assert_type(c, Operator[torch.Tensor, tuple[torch.Tensor]])


def test_sum_operator_tensor():
    rng = RandomGenerator(0)
    a = DummyOperator(torch.tensor(3.0))
    b = rng.complex64_tensor((3, 10))
    c = a + b
    x = rng.complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] + b * x

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator + tensor should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator + tensor should not be a LinearOperator'
    assert_type(c, Operator[torch.Tensor, tuple[torch.Tensor]])


def test_rsum_operator_tensor():
    rng = RandomGenerator(0)
    a = DummyOperator(torch.tensor(3.0))
    b = rng.complex64_tensor((3, 10))
    c = cast(Operator, b + a)  # required due to https://github.com/pytorch/pytorch/issues/124015
    x = rng.complex64_tensor(10)
    (y1,) = c(x)
    y2 = a(x)[0] + b * x

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, Operator), 'Operator + tensor should be an Operator'
    assert not isinstance(c, LinearOperator), 'Operator + tensor should not be a LinearOperator'


def test_sum_linearoperator_tensor():
    rng = RandomGenerator(0)
    a = DummyLinearOperator(rng.complex64_tensor((3, 3)))
    b = rng.complex64_tensor((3,))
    c = a + b
    x = rng.complex64_tensor(3)
    (y1,) = c(x)
    y2 = a(x)[0] + b * x

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, LinearOperator), 'LinearOperator + tensor should be a LinearOperator'
    assert isinstance(c, Operator), 'LinearOperator + tensor should be an Operator'


def test_rsum_linearoperator_tensor():
    rng = RandomGenerator(0)
    a = DummyLinearOperator(rng.complex64_tensor((3, 3)))
    b = rng.complex64_tensor((3,))
    c = cast(LinearOperator, b + a)  # required due to https://github.com/pytorch/pytorch/issues/124015
    x = rng.complex64_tensor(3)
    (y1,) = c(x)
    y2 = a(x)[0] + b * x

    torch.testing.assert_close(y1, y2)
    assert isinstance(c, LinearOperator), 'LinearOperator + tensor should be a LinearOperator'
    assert isinstance(c, Operator), 'LinearOperator + tensor should be an Operator'


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
    c = cast(DummyOperator, b * a)
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
    linear_op_product = cast(LinearOperator, b * a)  # required due to https://github.com/pytorch/pytorch/issues/124015
    dotproduct_adjointness_test(linear_op_product, u, v)


def test_adjoint_sum():
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    u = rng.complex64_tensor(10)
    v = rng.complex64_tensor(3)
    linear_op_sum = a + b
    dotproduct_adjointness_test(linear_op_sum, u, v)


def test_adjoint_tensor_sum():
    rng = RandomGenerator(3)
    a = DummyLinearOperator(rng.complex64_tensor((3, 3)))
    b = rng.float32_tensor(3)
    u = rng.complex64_tensor(3)
    v = rng.complex64_tensor(3)
    linear_op_sum = a + b
    dotproduct_adjointness_test(linear_op_sum, u, v)


def test_sum_operator_multiple():
    a = DummyOperator(torch.tensor(2.0))
    op_sum = a + (a + a) + (a + a) + a
    assert len(op_sum._operators) == 6
    x = RandomGenerator(0).complex64_tensor(10)
    (actual,) = op_sum(x)
    expected = 6 * a(x)[0]
    torch.testing.assert_close(actual, expected)


def test_sum_operator_multiple_adjoint():
    rng = RandomGenerator(7)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    linear_op_sum = a + (a + a) + (a + a) + a
    assert len(linear_op_sum._operators) == 6
    u = rng.complex64_tensor(10)
    v = rng.complex64_tensor(3)
    dotproduct_adjointness_test(linear_op_sum, u, v)


def test_adjoint_of_adjoint():
    """Test that the adjoint of the adjoint is the original operator"""
    a = DummyLinearOperator(RandomGenerator(7).complex64_tensor((3, 10)))
    assert a.H.H is a


def test_gram_shortcuts():
    """Test that .gram for composition and scalar multiplication results in shortcuts."""

    class GramOnlyOperator(LinearOperator):
        """Operator-Wrapper that only has a working .gram property."""

        def __init__(self, op: LinearOperator):
            super().__init__()
            self.op = op

        def forward(self, _: torch.Tensor):
            raise RuntimeError('This operator should not be called')

        def adjoint(self, _: torch.Tensor):
            raise RuntimeError('This operator should not be called')

        @property
        def gram(self):
            return self.op.H @ self.op

    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = DummyLinearOperator(rng.complex64_tensor((10, 10)))
    a_gram_only = GramOnlyOperator(a)
    # ignore required due to https://github.com/pytorch/pytorch/issues/124015
    op: LinearOperator = torch.tensor(1) * ((3 + 4j) * a_gram_only @ b) * (1 + 2j)  # type: ignore[assignment]
    gram = op.gram

    u = rng.complex64_tensor(10)
    v = rng.complex64_tensor(10)

    # Next line would raise an error if forward or adjoint were called on a_gram_only,
    # but if all shortcuts are taken, it should work as only .gram is called
    dotproduct_adjointness_test(gram, u, v)


def test_gram_correctness():
    """Test that the gram property is numerically correct."""
    rng = RandomGenerator(2)
    a = DummyLinearOperator(rng.complex64_tensor((3, 10)))
    b = DummyLinearOperator(rng.complex64_tensor((10, 10)))
    op: Any = rng.complex64_tensor(3) * ((3 + 4j) * a @ b) * (1 + 2j) * rng.complex64_tensor(10)
    gram = op.gram
    u = rng.complex64_tensor(10)
    actual = gram(u)[0]
    expected = op.H(*op(u))[0]
    torch.testing.assert_close(actual, expected)
