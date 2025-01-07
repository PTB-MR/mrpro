from dataclasses import dataclass, field
from typing import assert_type

import pytest
from mrpro.utils import getnestedattr, getnesteditem


@dataclass
class C:
    """Test class for getnestedattr."""

    c: int = 1


@dataclass
class B:
    """Test class for getnestedattr."""

    b: C = field(default_factory=C)


@dataclass
class A:
    """Test class for getnestedattr."""

    a: B = field(default_factory=B)


def test_getnestedattr_value() -> None:
    """Test getnestedattr with a valid path."""
    obj = A()
    actual = getnestedattr(obj, 'a', 'b', 'c')
    assert actual == 1


def test_getnestedattr_default() -> None:
    """Test getnestedattr with a missing path and a default value."""
    obj = A()
    actual = getnestedattr(obj, 'a', 'doesnotexist', 'c', default=2)
    assert_type(actual, int)
    assert actual == 2


def test_getnestedattr_type() -> None:
    """Test getnestedattr with a missing path no default value, but a return type."""
    obj = A()
    actual = getnestedattr(obj, 'a', 'doesnotexist', 'c', return_type=int)
    assert_type(actual, int | None)
    assert actual is None


def test_getnestedattr_default_type_error() -> None:
    """Test getnestedattr with a default value and a return type that do not match."""
    obj = A()
    with pytest.raises(TypeError):
        getnestedattr(obj, 'a', default=2, return_type=str)


def test_getnesteditem_value() -> None:
    """Test getnesteditem with a valid path."""
    obj = {'a': {'b': {'c': 1}}}
    actual = getnesteditem(obj, 'a', 'b', 'c')
    assert actual == 1


def test_getnesteditem_default() -> None:
    """Test getnesteditem with a missing path and a default value."""
    obj = {'a': {'b': {'c': 1}}}
    actual = getnesteditem(obj, 'a', 'doesnotexist', 'c', default=2)
    assert_type(actual, int)
    assert actual == 2


def test_getnesteditem_type() -> None:
    """Test getnesteditem with a missing path no default value, but a return type."""
    obj = {'a': {'b': {'c': 1}}}
    actual = getnesteditem(obj, 'a', 'doesnotexist', 'c', return_type=int)
    assert_type(actual, int | None)
    assert actual is None


def test_getnesteditem_default_type_error() -> None:
    """Test getnesteditem with a default value and a return type that do not match."""
    obj = {'a': {'b': {'c': 1}}}
    with pytest.raises(TypeError):
        getnesteditem(obj, 'a', default=2, return_type=str)
