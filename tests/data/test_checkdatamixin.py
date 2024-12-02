from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pytest
import torch
from mrpro.data.CheckDataMixin import Annotation, CheckDataMixin, ShapeError, SuspendDataChecks


def tuple_to_regex(t: tuple) -> str:
    """Convert a tuple to a regex-compatible string with escaped parentheses."""
    elements = ', '.join(map(str, t))
    if len(t) == 1:
        elements += ','  # Add trailing comma for single-element tuples
    return f'\\({elements}\\)'


@dataclass
class CheckedDataClass(CheckDataMixin):
    """A test dataclass."""

    float_tensor: Annotated[
        torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')
    ]
    int_tensor: Annotated[torch.Tensor, Annotation(dtype=(torch.int), shape='*#other #coil=1 #k2 #k1 k0=1')]
    fixed_ndarray: Annotated[np.ndarray, Annotation(dtype=int, shape='k0')]
    string: str


@dataclass(slots=True)
class Slots(CheckDataMixin):
    """A test dataclass with slots"""

    float_tensor: Annotated[
        torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')
    ]


@dataclass(frozen=True)
class Frozen(CheckDataMixin):
    """A frozen test dataclass"""

    float_tensor: Annotated[
        torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other coil #k2 #k1 #k0')
    ]


def test_checked_dataclass_success():
    n_coil = 2
    n_k2 = 3
    n_k1 = 3
    n_k0 = 4
    n_other = (1, 2, 3)
    _ = CheckedDataClass(
        float_tensor=torch.ones(*n_other, n_coil, n_k2, n_k1, n_k0),
        int_tensor=torch.zeros(*(1,), 1, n_k2, 1, 1, dtype=torch.int),
        fixed_ndarray=np.zeros(n_k0, dtype=int),
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
            fixed_ndarray=np.zeros(n_k0, dtype=int),
            string='test',
        )


def test_suspend_check_success():
    """Test the SuspendDataChecks context with a valid shape on exit"""
    with SuspendDataChecks():
        instance = Slots(torch.zeros(1))
        # fix the shape
        instance.float_tensor = torch.zeros(1, 1, 1, 1, 1)


def test_suspend_check_fail():
    """Test the SuspendDataChecks context with an invalid shape on exit"""
    with pytest.raises(ShapeError, match='dimensions'), SuspendDataChecks():
        _ = Slots(torch.zeros(1))


# TODO:
# string to size
# no dtype, object not having dtype
# no shape, object ithout shape

# shape property
# dtype property


# _ und ...
