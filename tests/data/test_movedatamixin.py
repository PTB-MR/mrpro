"""Tests the MoveDataMixin class."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import pytest
import torch
from mrpro.data import MoveDataMixin


class SharedModule(torch.nn.Module):
    """A module with two submodules that share the same parameters."""

    def __init__(self):
        super().__init__()
        self.module1 = torch.nn.Linear(1, 1)
        self.module2 = torch.nn.Linear(1, 1)
        self.module2.weight = self.module1.weight


@dataclass(slots=True)
class A(MoveDataMixin):
    """Test class A."""

    floattensor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    complextensor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.complex64))
    inttensor: torch.Tensor = field(default_factory=lambda: torch.tensor(1, dtype=torch.int32))
    booltensor: torch.Tensor = field(default_factory=lambda: torch.tensor(True))
    module: torch.nn.Module = field(default_factory=lambda: torch.nn.Linear(1, 1))


@dataclass(frozen=True)
class B(MoveDataMixin):
    """Test class B."""

    child: A = field(default_factory=A)
    module: torch.nn.Module = field(default_factory=SharedModule)
    floattensor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    complextensor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.complex64))
    inttensor: torch.Tensor = field(default_factory=lambda: torch.tensor(1, dtype=torch.int32))
    booltensor: torch.Tensor = field(default_factory=lambda: torch.tensor(True))


def _test(
    original: Any,
    new: Any,
    attribute: str,
    copy: bool,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    """Assertion used in the tests below.

    Compares the attribute of the original and new object.
    Checks device, dtype and if the data is copied if required
    on the new object.
    """
    original_data = getattr(original, attribute)
    new_data = getattr(new, attribute)
    if copy:
        assert new_data is not original_data, 'copy requested but not performed'
    assert torch.equal(new_data, original_data.to(device=expected_device, dtype=expected_dtype))
    assert new_data.device == expected_device, 'device not set correctly'
    assert new_data.dtype == expected_dtype, 'dtype not set correctly'


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('copy', [True, False])
def test_movedatamixin_float64like(copy: bool, dtype: torch.dtype):
    original = B()
    new = original.to(dtype=dtype, copy=copy)

    # Tensor attributes
    def test(attribute, expected_dtype):
        return _test(original, new, attribute, copy, expected_dtype, torch.device('cpu'))

    test('floattensor', torch.float64)
    test('complextensor', torch.complex128)
    test('inttensor', torch.int32)
    test('booltensor', torch.bool)

    # Attributes of child
    def testchild(attribute, expected_dtype):
        return _test(original.child, new.child, attribute, copy, expected_dtype, torch.device('cpu'))

    testchild('floattensor', torch.float64)
    testchild('complextensor', torch.complex128)
    testchild('inttensor', torch.int32)
    testchild('booltensor', torch.bool)

    # Module attribute
    _test(original.child.module, new.child.module, 'weight', copy, torch.float64, torch.device('cpu'))

    # No-copy required for these
    if not copy:
        assert original.inttensor is new.inttensor, 'no copy of inttensor required'
        assert original.booltensor is new.booltensor, 'no copy of booltensor required'
        assert original.child.inttensor is new.child.inttensor, 'no copy of inttensor required'
        assert original.child.booltensor is new.child.booltensor, 'no copy of booltensor required'
    assert original is not new, 'original and new should not be the same object'

    assert new.module.module1.weight is new.module.module1.weight, 'shared module parameters should remain shared'
