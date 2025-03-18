"""Tests the move to device/dtype functionality of  dataclasses."""

from dataclasses import field

import pytest
import torch
from mrpro.data import Dataclass, Rotation, SpatialDimension
from typing_extensions import Any

from tests import RandomGenerator


class SharedModule(torch.nn.Module):
    """A module with two submodules that share the same parameters."""

    def __init__(self):
        super().__init__()
        self.module1 = torch.nn.Linear(1, 1)
        self.module2 = torch.nn.Linear(1, 1)
        self.module2.weight = self.module1.weight


class A(Dataclass):
    """Test class A."""

    floattensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(0).float32_tensor((1, 20)))
    floattensor2: torch.Tensor = field(default_factory=lambda: RandomGenerator(1).float32_tensor((10, 1)))
    complextensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(2).complex64_tensor((1, 1)))
    inttensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(3).int32_tensor((10, 20)))
    booltensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(4).bool_tensor((10, 20)))
    module: torch.nn.Module = field(default_factory=lambda: torch.nn.Linear(1, 1))
    rotation: Rotation = field(default_factory=lambda: Rotation.random((10, 20), 0))


class B(Dataclass):
    """Test class B."""

    child: A = field(default_factory=A)
    module: SharedModule = field(default_factory=SharedModule)
    floattensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(0).float32_tensor((10, 20)))
    complextensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(1).complex64_tensor((1, 1)))
    inttensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(2).int32_tensor((10, 20)))
    booltensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(3).bool_tensor((10, 1)))
    doubletensor: torch.Tensor = field(default_factory=lambda: RandomGenerator(4).float64_tensor((1, 20)))


def _assert_attribute_properties(
    original: Any,
    new: Any,
    attribute: str,
    copy: bool,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    """Assertion used in the move to device/dtype tests.

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
def test_dataclass_to(copy: bool, dtype: torch.dtype) -> None:
    """Test dataclass.to using a nested object."""
    original = B()
    new = original.to(dtype=dtype, copy=copy)

    # Tensor attributes
    def test(attribute: str, expected_dtype: torch.dtype) -> None:
        return _assert_attribute_properties(original, new, attribute, copy, expected_dtype, torch.device('cpu'))

    test('floattensor', dtype.to_real())
    test('doubletensor', dtype.to_real())
    test('complextensor', dtype.to_complex())
    test('inttensor', torch.int32)
    test('booltensor', torch.bool)

    # Attributes of child
    def testchild(attribute: str, expected_dtype: torch.dtype) -> None:
        return _assert_attribute_properties(
            original.child, new.child, attribute, copy, expected_dtype, torch.device('cpu')
        )

    testchild('floattensor', dtype.to_real())
    testchild('complextensor', dtype.to_complex())
    testchild('inttensor', torch.int32)
    testchild('booltensor', torch.bool)

    # Module attribute
    _assert_attribute_properties(
        original.child.module, new.child.module, 'weight', copy, dtype.to_real(), torch.device('cpu')
    )

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
def test_dataclass_convert(copy: bool, dtype: torch.dtype, attribute: str) -> None:
    """Test dataclass.half/double/single using a nested object."""
    original = B()
    new = getattr(original, attribute)(copy=copy)

    # Tensor attributes
    def test(attribute, expected_dtype):
        return _assert_attribute_properties(original, new, attribute, copy, expected_dtype, torch.device('cpu'))

    test('floattensor', dtype.to_real())
    test('doubletensor', dtype.to_real())
    test('complextensor', dtype.to_complex())
    test('inttensor', torch.int32)
    test('booltensor', torch.bool)

    # Attributes of child
    def testchild(attribute, expected_dtype):
        return _assert_attribute_properties(
            original.child, new.child, attribute, copy, expected_dtype, torch.device('cpu')
        )

    testchild('floattensor', dtype.to_real())
    testchild('complextensor', dtype.to_complex())
    testchild('inttensor', torch.int32)
    testchild('booltensor', torch.bool)

    # Module attribute
    _assert_attribute_properties(
        original.child.module, new.child.module, 'weight', copy, dtype.to_real(), torch.device('cpu')
    )

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
def test_dataclass_cuda(already_moved: bool, copy: bool) -> None:
    """Test dataclass.cuda using a nested object."""
    original = B()
    if already_moved:
        original = original.cuda(torch.cuda.current_device())
    new = original.cuda(copy=copy)
    expected_device = torch.device(torch.cuda.current_device())
    assert new.device == expected_device

    # Tensor attributes
    def test(attribute, expected_dtype):
        return _assert_attribute_properties(original, new, attribute, copy, expected_dtype, expected_device)

    # all tensors should be of the same dtype as before
    test('floattensor', torch.float32)
    test('doubletensor', torch.float64)

    test('complextensor', torch.complex64)
    test('inttensor', torch.int32)
    test('booltensor', torch.bool)

    # Attributes of child
    def testchild(attribute, expected_dtype):
        return _assert_attribute_properties(original.child, new.child, attribute, copy, expected_dtype, expected_device)

    testchild('floattensor', torch.float32)
    testchild('complextensor', torch.complex64)
    testchild('inttensor', torch.int32)
    testchild('booltensor', torch.bool)

    # Module attribute
    _assert_attribute_properties(
        original.child.module, new.child.module, 'weight', copy, torch.float32, expected_device
    )

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


def test_dataclass_apply_() -> None:
    """Tests apply_ method of dataclass."""
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


def test_dataclass_apply() -> None:
    """Tests apply method of dataclass."""
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


def test_dataclass_no_new_attributes() -> None:
    """Tests that new attributes are not added to the dataclass."""
    data = B()
    with pytest.raises(AttributeError):
        data.doesnotexist = 1  # type: ignore[attr-defined]


def check_broadcastable(actual_shape, expected_shape):
    """Raise a Runtime Error is actual is not boradcastable to expected."""
    torch.empty(actual_shape).broadcast_to(expected_shape)


@pytest.mark.parametrize(
    ('index', 'expected_shape'),
    [
        (0, (1, 20)),
        (slice(0, 5), (5, 20)),
        ((Ellipsis, (2, 3)), (10, 2)),
        (((0, 1), (0, 1)), (2, 1, 1)),
    ],
    ids=['single', 'slice', 'ellipsis', 'vectorized'],
)
def test_dataclass_getitem(index, expected_shape: tuple[int, ...]) -> None:
    """Test the __getitem__ method of the dataclass."""
    # The indexing itself is alreadytested in test_indexer.py
    # Thus, this test only needs to check that the indexing if performed on the attributes.
    indexed = B()[index]
    check_broadcastable(indexed.floattensor.shape, expected_shape)
    check_broadcastable(indexed.child.floattensor.shape, expected_shape)
    check_broadcastable(indexed.child.rotation.shape, expected_shape)
    check_broadcastable(indexed.child.shape, expected_shape)
    check_broadcastable(indexed.shape, expected_shape)


def test_dataclass_reducerepeat() -> None:
    """Test reduction of repeated dimensions."""

    class Container(Dataclass):
        a: torch.Tensor
        b: SpatialDimension
        c: Rotation

    rng = RandomGenerator(10)

    a = rng.float32_tensor((5, 1, 1, 1))
    a_expanded = a.expand(5, 2, 3, 1)

    b = SpatialDimension(*rng.float32_tensor((3, 1, 1, 3)))
    b_expanded = SpatialDimension(*[x.expand(1, 2, 3) for x in b.zyx])

    c_matrix = torch.eye(3).reshape(1, 1, 3, 3)
    c_expanded = Rotation.from_matrix(c_matrix.expand(5, 2, 3, 3))

    test = Container(a_expanded, b_expanded, c_expanded)

    torch.testing.assert_close(test.a, a)
    torch.testing.assert_close(test.b.z, b.z)
    torch.testing.assert_close(test.b.y, b.y)
    torch.testing.assert_close(test.b.x, b.x)
    torch.testing.assert_close(test.c.as_matrix(), c_matrix)
