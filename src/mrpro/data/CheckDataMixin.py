"""Dtype and shape checking for dataclasses."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Annotated, Literal, TypeAlias, get_args, get_origin

import torch
from typing_extensions import Protocol, Self, runtime_checkable


@dataclass(frozen=True)
class _Dim:
    """A dimension in a shape specification."""

    name: str
    broadcastable: bool


class _NamedDim(_Dim): ...


class _FixedDim(_Dim): ...


class _NamedVariadicDim(_Dim): ...


_anonymous_variadic_dim: Literal['...'] = '...'
_anonymous_dim: Literal['_'] = '_'
_DimType: TypeAlias = Literal['...', '_'] | _NamedDim | _NamedVariadicDim | _FixedDim


class RuntimeCheckError(Exception):
    """An error in the runtime checks data fields."""


class ShapeError(RuntimeCheckError):
    """A mismatch in the shape of a field."""


class DtypeError(RuntimeCheckError):
    """A mismatch in the dtype of a field."""


class SpecificationError(RuntimeCheckError):
    """A syntax error in the specification."""


class FieldTypeError(RuntimeCheckError):
    """A mismatch in the type of a field."""


@lru_cache
def _parse_string_to_shape_specification(dim_str: str) -> tuple[tuple[_DimType, ...], int | None]:
    """Parse a string representation of a shape specification."""
    if not isinstance(dim_str, str):
        raise SpecificationError('Shape specification must be a string.')
    dims: list[_DimType] = []
    index_variadic = None
    for index, elem in enumerate(dim_str.split()):
        if '...' in elem:
            if elem != '...':
                raise SpecificationError("Anonymous multiple axes '...' must be used on its own; " f'got {elem}')
            if index_variadic is not None:
                raise SpecificationError('Cannot use variadic specifiers (`*name` or `...`) ' 'more than once.')
            index_variadic = index
            dims.append(_anonymous_variadic_dim)
            continue

        broadcastable = '#' in elem
        elem = elem.replace('#', '')
        variadic = '*' in elem
        elem = elem.replace('*', '')
        anonymous = '_' in elem
        elem = elem.replace('_', '')
        elem = elem.split('=')[-1]
        fixed = len(elem) != 0 and not elem.isidentifier()

        if fixed:
            dims.append(_FixedDim(elem, broadcastable))
        elif variadic:
            dims.append(_NamedVariadicDim(elem, broadcastable))
            if index_variadic is not None:
                raise SpecificationError('Cannot use variadic specifiers (`*name` or `...`) ' 'more than once.')
            index_variadic = index
        elif anonymous:
            dims.append(_anonymous_dim)
        else:
            dims.append(_NamedDim(elem, broadcastable))

        if fixed and variadic:
            raise SpecificationError('Cannot have a fixed axis bind to multiple axes, e.g. `*4` is not allowed.')
        if fixed and anonymous:
            raise SpecificationError('Cannot have a fixed axis be anonymous, e.g. `_4` is not allowed.')
        if anonymous and broadcastable:
            raise SpecificationError(
                'Cannot have an axis be both anonymous and broadcastable, e.g. `#_` is not allowed.'
            )
    return tuple(dims), index_variadic


def _shape_specification_to_string(dims: tuple[_DimType, ...]) -> str:
    """Convert a shape specification to a string."""
    string = ''
    for dim in dims:
        match dim:
            case _FixedDim(name, broadcastable):
                string += f"{name}{'#' if broadcastable else ''} "
            case _NamedDim(name, broadcastable):
                string += f"{name}{'#' if broadcastable else ''} "
            case _NamedVariadicDim(name, broadcastable):
                string += f"*{name}{'#' if broadcastable else ''} "
            case _ if dim == _anonymous_dim:
                string += '_ '
            case _ if dim == _anonymous_variadic_dim:
                string += '... '
            case _:
                raise SpecificationError(f'Unknown dimension type {dim}')
    return string.strip()


class ShapeMemo(Mapping):
    """Immutable memoization object for shapes of named dimensions."""

    def __init__(self, *arg, **kwargs):
        """Initialize the memoization object."""
        self._d = dict(*arg, **kwargs)

    def __getitem__(self, key: str) -> tuple[tuple[int, ...], bool]:
        """Get the shape of a named dimension."""
        value = self._d[key]
        if isinstance(value, int):
            return (value,), False
        return value

    def __len__(self) -> int:
        """Get the number of named dimensions stored in the memo."""
        return len(self._d)

    def __or__(self, other: Self | dict) -> Self:
        """Combine two memoization objects."""
        if isinstance(other, dict):
            return type(self)(self._d | other)
        elif isinstance(other, ShapeMemo):
            return type(self)(self._d | other._d)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __iter__(self) -> Iterator:
        """Get an iterator over the named dimensions stored in the memo."""
        return iter(self._d)

    def __hash__(self):
        """Hash the memoization object.

        This is necessary to use the memoization object as a key in a dictionary or
        in a cache. The hash is based on the memory address of the object, i.e.
        each memoization object is unique.
        """
        return id(self)

    def __repr__(self) -> str:
        """Get a string representation of the memoization object."""
        return f'Memo({self._d})'


@lru_cache
def _check_dim(memo: ShapeMemo, dim: _DimType, size: int) -> ShapeMemo:
    """Check if a single dimension matches the memo. Return the updated memo."""
    if dim == _anonymous_variadic_dim:
        # should not happen?
        raise ShapeError('anonymous variadic dimensions should not be checked')
    elif dim == _anonymous_dim:
        # no need to check anonymous dimensions
        pass
    elif dim.broadcastable and size == 1:
        # will always match
        pass
    elif isinstance(dim, _FixedDim) and int(dim.name) != size:
        raise ShapeError(f'the dimension size {size} does not equal {dim.name} as expected by annotation')
    elif isinstance(dim, _NamedDim):
        try:
            prev = memo[dim.name][0][0]
        except KeyError:
            memo = ShapeMemo(memo, **{dim.name: size})
            return memo
        if prev != size:
            raise ShapeError(
                f'the size of dimension {dim.name} is {size} which does not equal the previous seen value of {prev}'
            )
    return memo


@lru_cache
def _check_named_variadic_dim(
    memo: ShapeMemo, variadic_dim: _NamedVariadicDim, variadic_shape: tuple[int, ...]
) -> ShapeMemo:
    """Check if named variadic dimension matches the memo. Return the updated memo."""
    name = variadic_dim.name
    broadcastable = variadic_dim.broadcastable
    try:
        prev_shape, prev_broadcastable = memo[name]
    except KeyError:
        # first time we see this variadic dimension, it will always match
        memo = ShapeMemo(memo, **{name: (variadic_shape, broadcastable)})
        return memo
    if prev_broadcastable or broadcastable:
        try:
            broadcast_shape = torch.broadcast_shapes(variadic_shape, prev_shape)
        except ValueError:  # not broadcastable
            raise ShapeError(
                f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, "
                f'which cannot be broadcast with the existing value of {prev_shape}'
            ) from None
        if not broadcastable and broadcast_shape != variadic_shape:
            raise ShapeError(
                f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, "
                f'which the existing value of {prev_shape} cannot be broadcast to'
            )
        if not prev_broadcastable and broadcast_shape != prev_shape:
            raise ShapeError(
                f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, "
                f'which cannot be broadcast to the existing value of {prev_shape}'
            )
        memo = ShapeMemo(memo, **{name: (variadic_shape, broadcastable)})
        return memo
    if variadic_shape != prev_shape:
        raise ShapeError(
            f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, "
            f'which does not equal the existing value of {prev_shape}'
        )
    return memo


@lru_cache
def _parse_string_to_size(shape_string: str, memo: ShapeMemo) -> tuple[int, ...]:
    """Convert a string representation of a shape to a shape."""
    shape: list[int] = []
    for dim in _parse_string_to_shape_specification(shape_string)[0]:
        if isinstance(dim, _FixedDim):
            shape.append(int(dim.name))
        elif isinstance(dim, _NamedDim):
            try:
                shape.append(memo[dim.name][0][0])
            except KeyError:
                if not dim.broadcastable:
                    raise KeyError(f'Dimension {dim.name} not found in memo') from None
                shape.append(1)
        elif isinstance(dim, _NamedVariadicDim):
            try:
                shape.extend(memo[dim.name][0])
            except KeyError:
                raise KeyError(f'Variadic dimension {dim.name} not found in memo') from None
        elif dim is _anonymous_dim:
            shape.append(1)
        elif dim is _anonymous_variadic_dim:
            raise SpecificationError('Cannot convert anonymous variadic dimension to a shape')
    return tuple(shape)


@runtime_checkable
class HasDtype(Protocol):
    """Has a dtype attribute."""

    dtype: torch.dtype


@runtime_checkable
class HasShape(Protocol):
    """Has a shape attribute."""

    shape: tuple[int, ...]


class HasShapeAndDtype(HasDtype, HasShape):
    """Has both a shape and a dtype attribute."""


@dataclass(slots=True, init=False)
class Annotation:
    """A shape and dtype annotation for a tensor-like object."""

    dtype: tuple[torch.dtype, ...] | None = None
    shape: tuple[_DimType, ...] | None = None
    index_variadic: int | None = None

    def __init__(self, dtype: torch.dtype | Sequence[torch.dtype] | None = None, shape: str | None = None) -> None:
        """Initialize the annotation.

        Parameters
        ----------
        dtype
            the acceptable dtype(s) for the object
        shape
            a string representation of the shape of the object.
            The string should be a space-separated list of dimensions.
            Each dimension can either be a fixed integer size (`5`) or a named dimension (`batch`),
            The dimension can be prefixed with `*` to indicate that they are variadic.
            The dimensions can be prefixed with `#` to indicate that they are broadcastable.
            The dimensions can be prefixed with `_` to ignore the size of the dimension in all checks.
            This can also be used on its own.
            Fixed dimensions can be prefixed with `string=`,
            were the string will be ignored and only serves as documentation.

            Example:
               `*#batch channel=2 depth #height #width` indicates that the object has at least  dimensions.
               The last two dimensions are named `height` and `width`.
               These must be broadcastable for all objects using the same memoization object.
               The depth dimensions must match exactly for all objects using the same memoization object.
               The channel dimension must be exactly 2.
               There can be any number of batch dimensions before the named dimensions, but across all objects
                using the same memoization object, their shape as to be broadcastable.
            For more information, see `jaxtyping <https://docs.kidger.site/jaxtyping/api/array/>`__
        """
        if shape is not None:
            self.shape, self.index_variadic = _parse_string_to_shape_specification(shape)
        if dtype is not None:
            if isinstance(dtype, torch.dtype):
                dtype = (dtype,)
            self.dtype = tuple(dtype)

    def assert_dtype(self, obj: HasDtype) -> None:
        """Raise a DtypeError if the object does not have the expected dtype.

        Parameters
        ----------
        obj
            the object to check
        """
        if self.dtype is None:
            return
        if not hasattr(obj, 'dtype'):
            raise TypeError('this object does not have a dtype attribute')
        if obj.dtype not in self.dtype:
            raise DtypeError(f'this object has dtype {obj.dtype}, not any of {self.dtype} as expected by the type hint')

    def __repr__(self) -> str:
        """Get a string representation of the annotation."""
        arguments = []
        if self.shape is not None:
            arguments.append(f"shape='{_shape_specification_to_string(self.shape) }'")
        if self.dtype is not None:
            arguments.append(f'dtype={self.dtype}')
        representation = f"Annotation({', '.join(arguments)})"
        return representation

    def assert_shape(self, obj: HasShape, memo: ShapeMemo | None = None) -> None | ShapeMemo:
        """Raise a ShapeError if the object does not have the expected shape.

        Parameters
        ----------
        obj
            the object to check
        memo
            a memoization object storing the shape of named dimensions from
            previous checks. Will not be modified.

        Returns
        -------
        memo
            a new memoization object storing the shape of named dimensions from
            previous and the current check.
        """
        if self.shape is None:
            return memo
        if not hasattr(obj, 'shape'):
            raise TypeError('this object does not have a shape attribute')
        if memo is None:
            memo = ShapeMemo()
        if self.index_variadic is None:
            # no variadic dimension
            if len(obj.shape) != len(self.shape):
                raise ShapeError(
                    f'this array has {len(obj.shape)} dimensions, not the {len(self.shape)} expected by the type hint'
                )
            for dim, size in zip(self.shape, obj.shape, strict=False):
                memo = _check_dim(memo, dim, size)
            return memo
        if len(obj.shape) < len(self.shape) - 1:
            raise ShapeError(
                f'this array has {len(obj.shape)} dimensions, which is fewer than {len(self.shape) - 1} '
                'that is the minimum expected by the type hint'
            )
        index_variadic_end = -(len(self.shape) - self.index_variadic - 1)
        pre_variadic_shape = obj.shape[: self.index_variadic]
        post_variadic_shape = obj.shape[index_variadic_end:]

        for dim, size in zip(self.shape[: self.index_variadic], pre_variadic_shape, strict=False):
            memo = _check_dim(memo, dim, size)  # dims before variadic
        for dim, size in zip(self.shape[self.index_variadic + 1 :], post_variadic_shape, strict=False):
            memo = _check_dim(memo, dim, size)  # dims after variadic

        variadic_dim = self.shape[self.index_variadic]
        variadic_shape = obj.shape[self.index_variadic : index_variadic_end]

        if variadic_dim == _anonymous_variadic_dim:
            # no need to check an anonymous variadic dimension further
            return memo
        assert isinstance(variadic_dim, _NamedVariadicDim)  # noqa:S101 # mypy hint
        memo = _check_named_variadic_dim(memo, variadic_dim, variadic_shape)
        return memo

    def check(self, obj: object, /, strict: bool = False, memo: ShapeMemo | None = None) -> ShapeMemo | None:
        """Check that an object satisfies the type hint.

        Parameters
        ----------
        obj
            the object to check
        strict
            if False, only check the shape if the object has a shape attribute
            and only check the dtype if the object has a dtype attribute
            if True, raise an Exception if the type hint specifies a dtype or shape
            but the object does not have the corresponding attribute
        memo
            a memoization object storing the shape of named dimensions from
            previous checks. Will not be modified.

        Returns
        -------
        memo
            a new memoization object storing the shape of named dimensions from
            previous and the current check.

        """
        if self.dtype is not None:
            if isinstance(obj, HasDtype):
                self.assert_dtype(obj)
            elif strict:
                raise SpecificationError('this object does not have a dtype attribute')
        if self.shape is not None:
            if isinstance(obj, HasShape):
                memo = self.assert_shape(obj, memo)
            elif strict:
                raise SpecificationError('this object does not have a shape attribute')
        return memo


SpecialTensor = Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other 1 #k2 #k1 1')]


@dataclass
class CheckDataMixin:
    """A Mixin to provide shape and dtype checking to dataclasses.

    If fields in the dataclass are hinted with an annotated type, for example
    `Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other 1 #k2 #k1 1')]`
    dtype and shape of the field will be checked after creation and whenever `check_invariants `is called.

    To pass the checks the following have rules have to be followed by each field:

    The dtype of a field has to be in the tuple of dtypes listed in the annotation.
    Use None to allow all dtypes.

    The shape of a field has to match the string specification.
    Named dimensions have to be the same over all annotated fields.
    Use `#` to allow broadcasting, `*` to capture variadic number of axes,
    `_` for unchecked dimensions and `...` for variadic unchecked dimensions.
    See :class:`Annotation` for more details.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the dataclass."""
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the dataclass."""
        raise NotImplementedError

    __slots__ = ('_memo',)

    def __init_subclass__(cls):
        """Initialize a checked data subclass."""
        # inject the check_invariants method into post_init
        original_post_init = vars(cls).get('__post_init__')

        def new_post_init(self: CheckDataMixin) -> None:
            self.check_invariants()
            if original_post_init:
                original_post_init(self)

        cls.__post_init__ = new_post_init  # type: ignore[attr-defined]

    def check_invariants(self) -> None:
        """Check that the dataclass invariants are satisfied."""
        memo = getattr(self, '_memo', ShapeMemo())
        for elem in fields(self):
            name = elem.name
            expected_type = elem.type
            value = getattr(self, name)
            if get_origin(expected_type) is Annotated:
                expected_type, *annotations = get_args(expected_type)
            else:
                annotations = []
            if not isinstance(expected_type, type):
                raise TypeError(
                    f'Expected a type, got {type(expected_type)}. This could be caused by __future__.annotations'
                )
            if not isinstance(value, expected_type):
                raise FieldTypeError(f'Expected {expected_type} for {name}, got {type(value)}')
            for annotation in annotations:
                # there could be other annotations not related to the shape and dtype
                if isinstance(annotation, Annotation):
                    try:
                        memo = annotation.check(value, memo=memo, strict=True)
                    except RuntimeCheckError as e:
                        raise type(e)(
                            f'Dataclass invariant violated for {self.__class__.__name__}.{name}: {e}\n {annotation}.'
                        ) from None
        object.__setattr__(self, '_memo', memo)  # works for frozen dataclasses


if __name__ == '__main__':
    # Test the CheckDataMixin
    @dataclass
    class Test(CheckDataMixin):
        """A test dataclass."""

        a: Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other 1 #k2 #k1 1')]
        b: Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other 1 #k2 #k1 1')]

    test1 = Test(torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1))
    test2 = Test(torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 2))
