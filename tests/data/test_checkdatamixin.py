from dataclasses import dataclass
from typing import Annotated

import pytest
import torch
from mrpro.data.CheckDataMixin import (
    Annotation,
    CheckDataMixin,
    ShapeError,
    SuspendDataChecks,
    _FixedDim,
    _NamedDim,
    _parse_string_to_shape_specification,
    _parse_string_to_size,
    _shape_specification_to_string,
)


def tuple_to_regex(t: tuple) -> str:
    """Convert a tuple to a regex-compatible string with escaped parentheses."""
    elements = ', '.join(map(str, t))
    if len(t) == 1:
        elements += ','  # Add trailing comma for single-element tuples
    return f'\\({elements}\\)'


# Tests that these can be defined
@dataclass
class CheckedDataClass(CheckDataMixin):
    """A test dataclass."""

    float_tensor: Annotated[
        torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')
    ]
    int_tensor: Annotated[torch.Tensor, Annotation(dtype=torch.int, shape='*#other #coil=1 #k2 #k1 k0=1')]
    string: str


@dataclass(slots=True)
class Slots(CheckDataMixin):
    """A test dataclass with slots"""

    tensor: Annotated[torch.Tensor, Annotation(shape='... _ 5 dim')]


@dataclass(frozen=True)
class Frozen(CheckDataMixin):
    """A frozen test dataclass"""

    tensor: Annotated[torch.Tensor, Annotation(dtype=(torch.float32,))]


def test_checked_dataclass_success():
    n_coil = 2
    n_k2 = 3
    n_k1 = 3
    n_k0 = 4
    n_other = (1, 2, 3)
    _ = CheckedDataClass(
        float_tensor=torch.ones(*n_other, n_coil, n_k2, n_k1, n_k0),
        int_tensor=torch.zeros(*(1,), 1, n_k2, 1, 1, dtype=torch.int),
        string='test',
    )


def test_checked_dataclass_variadic_fail():
    n_coil = 2
    n_k2 = 3
    n_k1 = 3
    n_k0 = 4
    n_other = (1, 2, 3)
    n_other_fail = (2,)
    with pytest.raises(
        ShapeError,
        match=f"'*other' is {tuple_to_regex(n_other_fail)}, "
        f'which cannot be broadcast with the existing value of {tuple_to_regex(n_other)}',
    ):
        _ = CheckedDataClass(
            float_tensor=torch.ones(*n_other, n_coil, n_k2, n_k1, n_k0),
            int_tensor=torch.zeros(*n_other_fail, 1, n_k2, 1, 1, dtype=torch.int),
            string='test',
        )


def test_checked_dataclass_fixed_fail():
    n_coil = 2
    n_k2 = 3
    n_k1 = 3
    n_k0 = 4
    n_other = (1, 2, 3)
    not_one = 17
    with pytest.raises(ShapeError, match=f' the dimension size {not_one} does not equal 1'):
        _ = CheckedDataClass(
            float_tensor=torch.ones(*n_other, n_coil, n_k2, n_k1, n_k0),
            int_tensor=torch.zeros(*n_other, 1, n_k2, 1, not_one, dtype=torch.int),
            string='test',
        )


def test_suspend_check_success():
    """Test the SuspendDataChecks context with a valid shape on exit"""
    with SuspendDataChecks():
        instance = Slots(torch.zeros(1))
        # fix the shape
        instance.tensor = torch.zeros(2, 3, 4, 5, 6)


def test_suspend_check_fail():
    """Test the SuspendDataChecks context with an invalid shape on exit"""
    with pytest.raises(ShapeError, match='dimensions'), SuspendDataChecks():
        _ = Slots(torch.zeros(1))


def test_shape():
    """Test the shape property"""
    n_coil = 2
    n_k2 = 3
    n_k1 = 4
    n_k0 = 5
    n_other = (1, 2)
    instance = CheckedDataClass(
        float_tensor=torch.ones(n_coil, n_k2, n_k1, n_k0),
        int_tensor=torch.zeros(*n_other, 1, n_k2, 1, 1, dtype=torch.int),
        string='test',
    )
    assert instance.shape == (*n_other, n_coil, n_k2, n_k1, n_k0)


def test_dype():
    """Test the dtype property"""
    instance = CheckedDataClass(
        float_tensor=torch.ones(3, 4, 5, 6, dtype=torch.float64),
        int_tensor=torch.zeros(2, 1, 4, 1, 1, dtype=torch.int),
        string='test',
    )
    # the result dtype of int and float64 is float54
    assert instance.dtype == torch.float64


@pytest.mark.parametrize(
    ('string', 'expected'),
    [
        ('1 comment=2', ((_FixedDim('1', False), _FixedDim('2', False)), None)),
        ('#name', ((_NamedDim('name', True),), None)),
        ('_', (('_',), None)),
        ('...', (('...',), 0)),
    ],
    ids=['fixed', 'named broadcastable', 'anonymous', 'anonymous variadic'],
)
def test_parse_shape(string: str, expected: tuple) -> None:
    parsed = _parse_string_to_shape_specification(string)
    assert parsed == expected


@pytest.mark.parametrize(
    ('expected', 'shape'),
    [
        ('1 2', ((_FixedDim('1', False), _FixedDim('2', False)))),
        ('#name', ((_NamedDim('name', True),))),
        ('_', (('_',))),
        ('...', (('...',))),
    ],
    ids=['fixed', 'named broadcastable', 'anonymous', 'anonymous variadic'],
)
def test_specification_to_string(expected: str, shape: tuple) -> None:
    """Test conversion of parsed specification back to a string"""
    string = _shape_specification_to_string(shape)
    assert string == expected


def test_string_to_shape():
    """Test conversion of string to shape"""
    instance = Frozen(torch.zeros(1, 2, 5, 2))  # has shape hint '... _ 5 dim'
    instance.check_invariants()
    shape = _parse_string_to_size('fixed=3 dim 1', instance._memo)  # type:ignore[attr-defined]
    assert shape == (3, 2, 1)
