"""Tests the MoveDataMixin class."""

from dataclasses import dataclass, field

import pytest
import torch
from mrpro.data import MoveDataMixin
from typing_extensions import Any


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
    floattensor2: torch.Tensor = field(default_factory=lambda: torch.tensor(-1.0))
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
    doubletensor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0, dtype=torch.float64))


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

    assert new_data.device == expected_device, 'device not set correctly'
    assert new_data.dtype == expected_dtype, 'dtype not set correctly'
    if torch.is_complex(original_data):
        # torch.equal not yet implemented for complex half tensors.
        assert torch.equal(
            torch.view_as_real(new_data),
            torch.view_as_real(original_data.to(device=expected_device, dtype=expected_dtype)),
        )
    else:
        assert torch.equal(new_data, original_data.to(device=expected_device, dtype=expected_dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64, torch.float64, torch.complex128])
@pytest.mark.parametrize('copy', [True, False])
def test_movedatamixin_to(copy: bool, dtype: torch.dtype):
    """Test MoveDataMixin.to using a nested object."""
    original = B()
    new = original.to(dtype=dtype, copy=copy)

    # Tensor attributes
    def test(attribute, expected_dtype):
        return _test(original, new, attribute, copy, expected_dtype, torch.device('cpu'))

    test('floattensor', dtype.to_real())
    test('doubletensor', dtype.to_real())
    test('complextensor', dtype.to_complex())
    test('inttensor', torch.int32)
    test('booltensor', torch.bool)

    # Attributes of child
    def testchild(attribute, expected_dtype):
        return _test(original.child, new.child, attribute, copy, expected_dtype, torch.device('cpu'))

    testchild('floattensor', dtype.to_real())
    testchild('complextensor', dtype.to_complex())
    testchild('inttensor', torch.int32)
    testchild('booltensor', torch.bool)

    # Module attribute
    _test(original.child.module, new.child.module, 'weight', copy, dtype.to_real(), torch.device('cpu'))

    # No-copy required for these
    if not copy:
        assert original.inttensor is new.inttensor, 'no copy of inttensor required'
        assert original.booltensor is new.booltensor, 'no copy of booltensor required'
        assert original.child.inttensor is new.child.inttensor, 'no copy of inttensor required'
        assert original.child.booltensor is new.child.booltensor, 'no copy of booltensor required'
    assert original is not new, 'original and new should not be the same object'

    assert new.module.module1.weight is new.module.module1.weight, 'shared module parameters should remain shared'


@pytest.mark.filterwarnings('ignore:ComplexHalf:UserWarning')
@pytest.mark.parametrize(
    ('dtype', 'attribute'), [(torch.float16, 'half'), (torch.float32, 'single'), (torch.float64, 'double')]
)
@pytest.mark.parametrize('copy', [True, False])
def test_movedatamixin_convert(copy: bool, dtype: torch.dtype, attribute: str):
    """Test MoveDataMixin.half/double/single using a nested object."""
    original = B()
    new = getattr(original, attribute)(copy=copy)

    # Tensor attributes
    def test(attribute, expected_dtype):
        return _test(original, new, attribute, copy, expected_dtype, torch.device('cpu'))

    test('floattensor', dtype.to_real())
    test('doubletensor', dtype.to_real())
    test('complextensor', dtype.to_complex())
    test('inttensor', torch.int32)
    test('booltensor', torch.bool)

    # Attributes of child
    def testchild(attribute, expected_dtype):
        return _test(original.child, new.child, attribute, copy, expected_dtype, torch.device('cpu'))

    testchild('floattensor', dtype.to_real())
    testchild('complextensor', dtype.to_complex())
    testchild('inttensor', torch.int32)
    testchild('booltensor', torch.bool)

    # Module attribute
    _test(original.child.module, new.child.module, 'weight', copy, dtype.to_real(), torch.device('cpu'))

    # No-copy required for these
    if not copy:
        assert original.inttensor is new.inttensor, 'no copy of inttensor required'
        assert original.booltensor is new.booltensor, 'no copy of booltensor required'
        assert original.child.inttensor is new.child.inttensor, 'no copy of inttensor required'
        assert original.child.booltensor is new.child.booltensor, 'no copy of booltensor required'
    assert original is not new, 'original and new should not be the same object'

    assert new.module.module1.weight is new.module.module1.weight, 'shared module parameters should remain shared'


@pytest.mark.cuda
@pytest.mark.parametrize('already_moved', [True, False])
@pytest.mark.parametrize('copy', [True, False])
def test_movedatamixin_cuda(already_moved: bool, copy: bool):
    """Test MoveDataMixin.cuda using a nested object."""
    original = B()
    if already_moved:
        original = original.cuda(torch.cuda.current_device())
    new = original.cuda(copy=copy)
    expected_device = torch.device(torch.cuda.current_device())
    assert new.device == expected_device

    # Tensor attributes
    def test(attribute, expected_dtype):
        return _test(original, new, attribute, copy, expected_dtype, expected_device)

    # all tensors should be of the same dtype as before
    test('floattensor', torch.float32)
    test('doubletensor', torch.float64)

    test('complextensor', torch.complex64)
    test('inttensor', torch.int32)
    test('booltensor', torch.bool)

    # Attributes of child
    def testchild(attribute, expected_dtype):
        return _test(original.child, new.child, attribute, copy, expected_dtype, expected_device)

    testchild('floattensor', torch.float32)
    testchild('complextensor', torch.complex64)
    testchild('inttensor', torch.int32)
    testchild('booltensor', torch.bool)

    # Module attribute
    _test(original.child.module, new.child.module, 'weight', copy, torch.float32, expected_device)

    # No-copy required for these
    if not copy and already_moved:
        assert original.inttensor is new.inttensor, 'no copy of inttensor required'
        assert original.booltensor is new.booltensor, 'no copy of booltensor required'
        assert original.floattensor is new.floattensor, 'no copy of floattensor required'
        assert original.doubletensor is new.doubletensor, 'no copy of doubletensor required'
        assert original.child.complextensor is new.child.complextensor, 'no copy of complextensor required'
        assert original.child.inttensor is new.child.inttensor, 'no copy of inttensor required'
        assert original.child.booltensor is new.child.booltensor, 'no copy of booltensor required'
    assert original is not new, 'original and new should not be the same object'

    assert new.module.module1.weight is new.module.module1.weight, 'shared module parameters should remain shared'


def test_movedatamixin_apply_():
    """Tests apply_ method of MoveDataMixin."""
    data = B()
    # make one of the parameters shared to test memo behavior
    data.child.floattensor2 = data.child.floattensor
    original = data.clone()

    def multiply_by_2(obj):
        if isinstance(obj, torch.Tensor):
            return obj * 2
        return obj

    data.apply_(multiply_by_2)
    torch.testing.assert_close(data.floattensor, original.floattensor * 2)
    torch.testing.assert_close(data.child.floattensor2, original.child.floattensor2 * 2)
    assert data.child.floattensor is data.child.floattensor2, 'shared module parameters should remain shared'


def test_movedatamixin_apply():
    """Tests apply method of MoveDataMixin."""
    data = B()
    # make one of the parameters shared to test memo behavior
    data.child.floattensor2 = data.child.floattensor
    original = data.clone()

    def multiply_by_2(obj):
        if isinstance(obj, torch.Tensor):
            return obj * 2
        return obj

    new = data.apply(multiply_by_2)
    torch.testing.assert_close(data.floattensor, original.floattensor)
    torch.testing.assert_close(data.child.floattensor2, original.child.floattensor2)
    torch.testing.assert_close(new.floattensor, original.floattensor * 2)
    torch.testing.assert_close(new.child.floattensor2, original.child.floattensor2 * 2)
    assert data.child.floattensor is data.child.floattensor2, 'shared module parameters should remain shared'
    assert new is not data, 'new object should be different from the original'
