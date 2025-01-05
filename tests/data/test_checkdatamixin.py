from dataclasses import dataclass
from typing import Annotated, Optional, Union

import pytest
import torch
from mrpro.data.CheckDataMixin import (
    Annotation,
    CheckDataMixin,
    DtypeError,
    RuntimeCheckError,
    ShapeError,
    ShapeMemo,
    SpecificationError,
    SuspendDataChecks,
    _FixedDim,
    _NamedDim,
    string_to_shape_specification,
    string_to_size,
    shape_specification_to_string,
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

    tensor1: Annotated[torch.Tensor, Annotation(shape='dim')]
    tensor2: Annotated[torch.Tensor, Annotation(shape='... _ 5 dim')]


@dataclass(frozen=True)
class Frozen(CheckDataMixin):
    """A frozen test dataclass"""

    tensor1: Annotated[torch.Tensor, Annotation(dtype=(torch.float32,))]


@dataclass
class WithOptional(CheckDataMixin):
    """A dataclass with None-able fields"""

    tensor: torch.Tensor | None = None
    or_tensor: Annotated[
        torch.Tensor | None, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')
    ] = None
    optional_tensor: Annotated[
        Optional[torch.Tensor], Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')  # noqa: UP007
    ] = None
    union_tensor: Annotated[
        Union[None, torch.Tensor], Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')  # noqa: UP007
    ] = None
    integer: int | None = None
    outer_or_tensor: (
        Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')]
        | None
    ) = None
    outer_optional_tensor: Optional[  # noqa: UP007
        Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')]
    ] = None
    outer_union_tensor: Union[  # noqa: UP007
        None,
        Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')],
    ] = None


def test_slots() -> None:
    """Test dataclass with slots."""
    # Declaration of the dataclass is already the main test
    Slots(torch.zeros(2), torch.zeros(1, 5, 2))


def test_frozen() -> None:
    """Test frozen dataclass."""
    # Declaration of the dataclass is already the main test
    Frozen(torch.ones(1))


def test_optional() -> None:
    """Test dataclass with None-able attributes"""
    WithOptional()
    WithOptional(
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
        1,
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
    )


def test_optional_fail() -> None:
    """Test exceptions with dataclass with None-able attributes"""
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, torch.ones(1))
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, torch.ones(1))
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, torch.ones(1))
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, None, None, torch.ones(1))
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, None, None, None, torch.ones(1))
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, None, None, None, None, torch.ones(1))

    with pytest.raises(RuntimeCheckError):
        WithOptional('not a tensor')  # type:ignore[arg-type]
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, 'not a tensor')  # type:ignore[arg-type]
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, 'not a tensor')  # type:ignore[arg-type]
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, 'not a tensor')  # type:ignore[arg-type]
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, None, 'not an integer')  # type:ignore[arg-type]
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, None, None, 'not a tensor')  # type:ignore[arg-type]
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, None, None, None, 'not a tensor')  # type:ignore[arg-type]
    with pytest.raises(RuntimeCheckError):
        WithOptional(None, None, None, None, None, None, None, 'not a tensor')  # type:ignore[arg-type]


def test_checked_dataclass_success() -> None:
    """Test successful checked dataclass"""
    n_coil = 2
    n_k2 = 3
    n_k1 = 3
    n_k0 = 4
    n_other = (1, 2, 3)
    CheckedDataClass(
        float_tensor=torch.ones(*n_other, n_coil, n_k2, n_k1, n_k0),
        int_tensor=torch.zeros(*(1,), 1, n_k2, 1, 1, dtype=torch.int),
        string='test',
    )


def test_checked_dataclass_variadic_fail() -> None:
    """Test exception raised on wrong variadic size"""
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
        CheckedDataClass(
            float_tensor=torch.ones(*n_other, n_coil, n_k2, n_k1, n_k0),
            int_tensor=torch.zeros(*n_other_fail, 1, n_k2, 1, 1, dtype=torch.int),
            string='test',
        )


def test_checked_dataclass_fixed_fail() -> None:
    """Test exception raised on wrong fixed size"""
    n_coil = 2
    n_k2 = 3
    n_k1 = 3
    n_k0 = 4
    n_other = (1, 2, 3)
    not_one = 17
    with pytest.raises(ShapeError, match=f' the dimension size {not_one} does not equal 1'):
        CheckedDataClass(
            float_tensor=torch.ones(*n_other, n_coil, n_k2, n_k1, n_k0),
            int_tensor=torch.zeros(*n_other, 1, n_k2, 1, not_one, dtype=torch.int),
            string='test',
        )


def test_suspend_check_success() -> None:
    """Test the SuspendDataChecks context with a valid shape on exit"""
    with SuspendDataChecks():
        instance = Slots(torch.zeros(6), torch.zeros(1))
        # fix the shape
        instance.tensor2 = torch.zeros(2, 3, 4, 5, 6)


def test_suspend_check_fail():
    """Test the SuspendDataChecks context with an invalid shape on exit"""
    with pytest.raises(ShapeError, match='dimensions'), SuspendDataChecks():
        # needs to be assigned to exist after leaving the Suspend context
        _ = Slots(torch.zeros(1), torch.zeros(1))


def test_shape() -> None:
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


def test_dype() -> None:
    """Test the dtype property"""
    instance = CheckedDataClass(
        float_tensor=torch.ones(3, 4, 5, 6, dtype=torch.float64),
        int_tensor=torch.zeros(2, 1, 4, 1, 1, dtype=torch.int),
        string='test',
    )
    # the result dtype of int and float64 is float54
    assert instance.dtype == torch.float64


def test_dype_fail() -> None:
    """Test the dtype exception"""
    with pytest.raises(DtypeError):
        CheckedDataClass(
            float_tensor=torch.ones(3, 4, 5, 6, dtype=torch.int),
            int_tensor=torch.zeros(2, 1, 4, 1, 1, dtype=torch.int),
            string='wrong float_tensor',
        )
    with pytest.raises(DtypeError):
        CheckedDataClass(
            float_tensor=torch.ones(3, 4, 5, 6, dtype=torch.int),
            int_tensor=torch.zeros(2, 1, 4, 1, 1, dtype=torch.float32),
            string='wrong int_tensor',
        )


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
    """Test parsing of shape string"""
    parsed = string_to_shape_specification(string)
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
    string = shape_specification_to_string(shape)
    assert string == expected


def test_string_to_shape() -> None:
    """Test conversion of string to shape"""
    instance = Slots(torch.zeros(2), torch.zeros(1, 2, 5, 2))  # has shape hint '... _ 5 dim'
    instance.check_invariants()
    memo = instance._memo  # type:ignore[attr-defined]
    memo = memo | {'fromdict': 8}
    memo = memo | ShapeMemo(frommemo=9)
    shape = string_to_size('fromdict frommemo fixed=3 dim 1', memo)
    assert shape == (8, 9, 3, 2, 1)

    with pytest.raises(KeyError):
        string_to_size('doesnotexist', memo)
    with pytest.raises(KeyError):
        string_to_size('*doesnotexist', memo)
    with pytest.raises(SpecificationError):
        string_to_size('...', memo)
