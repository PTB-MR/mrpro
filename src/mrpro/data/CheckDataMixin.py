"""Dtype and shape checking for dataclasses."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Annotated, Literal, TypeAlias, get_args, get_origin

import torch
from typing_extensions import Protocol, Self, runtime_checkable


@dataclass(frozen=True)
class _Dim:
    name: str
    broadcastable: bool


class _NamedDim(_Dim): ...


class _FixedDim(_Dim): ...


class _NamedVariadicDim(_Dim): ...


_anonymous_variadic_dim = '...'
_anonymous_dim = '_'
_DimType: TypeAlias = Literal['...', '_'] | _NamedDim | _NamedVariadicDim | _FixedDim


class RuntimeCheckError(Exception):
    """An error in the runtime shape and dtype checks."""


class ShapeError(RuntimeCheckError):
    """A missmatch in the shape of a tensor."""


class DtypeError(RuntimeCheckError):
    """A missmatch in the dtype of a tensor."""


class SpecificationError(RuntimeCheckError):
    """A syntax error in the specification."""


@lru_cache
def parse_dims(dim_str: str) -> tuple[tuple[_DimType, ...], int | None]:
    if not isinstance(dim_str, str):
        raise SpecificationError('Shape specification must be a string. Axes should be separated with ' 'spaces.')
    dims: list[_DimType] = []
    index_variadic = None
    for index, elem in enumerate(dim_str.split()):
        if '...' in elem:
            if elem != '...':
                raise SpecificationError("Anonymous multiple axes '...' must be used on its own; " f'got {elem}')
            broadcastable = False
            variadic = True
            anonymous = True
            dim_type = _NamedVariadicDim
        else:
            broadcastable = '#' in elem
            elem = elem.replace('#', '')
            variadic = '*' in elem
            elem = elem.replace('*', '')
            anonymous = '_' in elem
            elem = elem.replace('_', '')
            elem = elem.split('=')[-1]

            if len(elem) == 0 or elem.isidentifier():
                if variadic:
                    dim_type = _NamedVariadicDim
                else:
                    dim_type = _NamedDim
            else:
                dim_type = _FixedDim

        if variadic:
            if index_variadic is not None:
                raise SpecificationError('Cannot use variadic specifiers (`*name` or `...`) ' 'more than once.')
            index_variadic = index
        if dim_type is _FixedDim:
            if variadic:
                raise SpecificationError('Cannot have a fixed axis bind to multiple axes, e.g. `*4` is not allowed.')
            if anonymous:
                raise SpecificationError('Cannot have a fixed axis be anonymous, e.g. `_4` is not allowed.')
        if anonymous and broadcastable:
            raise SpecificationError(
                'Cannot have an axis be both anonymous and broadcastable, e.g. `#_` is not allowed.'
            )
        if anonymous and _anonymous_variadic_dim:
            dim = _anonymous_variadic_dim
        elif anonymous:
            dim = _anonymous_dim
        else:
            dim = dim_type(elem, broadcastable)

        dims.append(dim)
    return tuple(dims), index_variadic


class ShapeMemo(Mapping):
    """Immutable memoization object for shapes of named dimensions."""

    def __init__(self, *arg, **kwargs):
        """Initialize the memoization object."""
        self._d = dict(*arg, **kwargs)

    def __getitem__(self, key: str) -> int | tuple[bool, tuple[int, ...]]:
        """Get the shape of a named dimension."""
        return self._d[key]

    def __len__(self) -> int:
        """Get the number of named dimensions stored in the memo."""
        return len(self._d)

    def __or__(self, other: Self | dict) -> Self:
        """Combine two memoization objects."""
        if isinstance(other, dict):
            return ShapeMemo(self._d | other)
        elif isinstance(other, ShapeMemo):
            return ShapeMemo(self._d | other._d)
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
    def parse_string(self, string: str) -> tuple[int, ...]:
        """Convert a string representation of a shape to a shape."""
        shape: list[int] = []
        for dim in parse_dims(string)[0]:
            if isinstance(dim, _FixedDim):
                shape.append(int(dim.name))
            elif isinstance(dim, _NamedDim):
                try:
                    shape.append(self[dim.name])
                except KeyError:
                    if not dim.broadcastable:
                        raise KeyError(f'Dimension {dim.name} not found in memo') from None
                    shape.append(1)
            elif isinstance(dim, _NamedVariadicDim):
                try:
                    shape.extend(self[dim.name][1])
                except KeyError:
                    raise KeyError(f'Variadic dimension {dim.name} not found in memo') from None
            elif dim is _anonymous_dim:
                shape.append(1)
            elif dim is _anonymous_variadic_dim:
                raise SpecificationError('Cannot convert anonymous variadic dimension to a shape')
        return tuple(shape)

    @lru_cache
    def check_dim(self, dim: _DimType, size: int) -> Self:
        """Check if a single dimension matches the memo."""
        if dim is _anonymous_dim or dim.broadcastable and size == 1:
            pass
        elif isinstance(dim, _FixedDim) and int(dim.name) != size:
            raise ShapeError(f'the dimension size {size} does not equal {dim.name} as expected by the type hint')
        elif isinstance(dim, _NamedDim):
            try:
                prev = self[dim.name]
            except KeyError:
                memo = ShapeMemo(self, **{dim.name: size})
                return memo
            if prev != size:
                raise ShapeError(
                    f'the size of dimension {dim.name} is {size} which does not equal the previous seen value of {prev}'
                )
        return self

    @lru_cache
    def check_named_variadic_dim(self, variadic_dim: _NamedVariadicDim, variadic_shape: tuple[int, ...]) -> Self:
        """Check if named variadic dimension matches the memo."""
        name = variadic_dim.name
        broadcastable = variadic_dim.broadcastable
        try:
            prev_shape, prev_broadcastable = self[name]
        except KeyError:
            # first time we see this variadic dimension, it will always match
            memo = ShapeMemo(self, **{name: (variadic_shape, broadcastable)})
            return memo
        if prev_broadcastable or broadcastable:
            try:
                broadcast_shape = torch.broadcast_shapes(variadic_shape, prev_shape)
            except ValueError:  # not broadcastable
                raise ShapeError(
                    f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, which cannot be broadcast with the existing value of {prev_shape}"
                ) from None
            if not broadcastable and broadcast_shape != variadic_shape:
                raise ShapeError(
                    f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, which the existing value of {prev_shape} cannot be broadcast to"
                )
            if not prev_broadcastable and broadcast_shape != prev_shape:
                raise ShapeError(
                    f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, which cannot be broadcast to the existing value of {prev_shape}"
                )
            memo = ShapeMemo(self, **{name: (variadic_shape, broadcastable)})
            return memo
        if variadic_shape != prev_shape:
            raise ShapeError(
                f"the shape of its variadic dimensions '*{variadic_dim.name}' is {variadic_shape}, which does not equal the existing value of {prev_shape}"
            )
        return memo


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
            Fixed dimensions can be prefixed with `string=`. The string will be ignored and only serves as documentation.

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
            self.shape, self.index_variadic = parse_dims(shape)
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
                memo = memo.check_dim(dim, size)
            return memo
        if len(obj.shape) < len(self.shape) - 1:
            raise ShapeError(
                f'this array has {len(obj.shape)} dimensions, which is fewer than {len(self.shaöe) - 1} that is the minimum expected by the type hint'
            )
        index_variadic_end = -(len(self.shape) - self.index_variadic - 1)
        pre_variadic_shape = obj.shape[: self.index_variadic]
        post_variadic_shape = obj.shape[index_variadic_end:]

        for dim, size in zip(self.shape[: self.index_variadic], pre_variadic_shape, strict=False):
            memo = memo.check_dim(dim, size)  # dims before variadic
        for dim, size in zip(self.shape[self.index_variadic + 1 :], post_variadic_shape, strict=False):
            memo = memo.check_dim(dim, size)  # dims after variadic

        variadic_dim = self.shape[self.index_variadic]
        variadic_shape = obj.shape[self.index_variadic : index_variadic_end]

        if variadic_dim == _anonymous_variadic_dim:
            # no need to check an anonymous variadic dimension further
            return memo
        assert isinstance(variadic_dim, _NamedVariadicDim)  # noqa:S101 # mypy hint
        memo = memo.check_named_variadic_dim(variadic_dim, variadic_shape)
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
        if self.dtype is not None and (strict or isinstance(obj, HasDtype)):
            self.assert_dtype(obj)
        if self.shape is not None and (strict or isinstance(obj, HasShape)):
            memo = self.assert_shape(obj, memo)
        return memo


SpecialTensor = Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other 1 #k2 #k1 1')]


class CheckDataMixin:
    """A Mixin to provide shape and dtype checking to dataclasses.

    If fields in the dataclass are hinted with an annotated typpe, for example
    `Annotated[torch.Tensor, Annotation(dtype=(torch.float32, torch.float64), shape='*#other 1 #k2 #k1 1')]`
    dtype and shape of the field will be checked after creation and whenever `check_invariants `is called.

    To pass the checks the following have rules have to be followed by each field:
    The dtype of a field has to be in the tuple of dtypes listed in the annotation. Use None to allow all dtypes.
    The shape of a field has to match the string specifcation. Named dimensions have to be the same over all annotated fields.
    Use `#` to allow broadcasting, `*` to capture variadic number of axes, `_` for unchecked dimensions and `...` for variadic
    unchecked dimensions. See :class:`Annotation` for more details.

    At class definition, you can supply a shape string. The shape of the overall dataclass, as returned by `instance.shape`
    will be obtained by substitution of the named axes determined during the invariants checks.

    Also, a dtype for the dataclass (to be returned by `instance.dtype`) can be specified, either as a dtype or as
    a string that will be parsed - for example `self.data.dtype` would result in returning the dtype of the data field.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Either overload shape or specify a shape-string when inheriting from CheckDataMixin"""
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        """Either overload dtype or specify a dtype when inheriting from CheckDataMixin"""
        raise NotImplementedError

    def __init_subclass__(cls, *, shape: str | None = None, dtype: torch.dtype | str | None = None, **kwargs):
        """Initialize a checked data subclass"""
        if hasattr(cls, '_memo'):
            return

        if not hasattr(cls, '__annotations__'):
            cls.__annotations__ = {}
        cls.__annotations__['_memo'] = ShapeMemo
        cls._memo = field(init=False, repr=False, default_factory=ShapeMemo)

        original_post_init = vars(cls).get('__post_init__')

        def new_post_init(self) -> None:
            self.check_invariants()
            if original_post_init:
                original_post_init(self)

        # Set the combined __post_init__ on the class
        cls.__post_init__ = new_post_init

        @property
        def shape_property(self) -> tuple[int, ...]:
            """The shape of the dataclass.

            Not all fields have to have exactly this shape internally.
            """
            formatted_shape = eval(f"f'{shape}'")
            return self._memo.parse_string(formatted_shape)

        @property
        def dtype_property(self) -> torch.dtype:
            """The dtype of the dataclass.

            Not all fields have to have this exact dtype.
            """
            if isinstance(dtype, str):
                try:
                    parsed_dtype = eval(f'{dtype}')
                except SyntaxError:
                    raise SpecificationError(f'Invalid dtype specification: {dtype}') from None
            else:
                parsed_dtype = dtype
            if not isinstance(parsed_dtype, torch.dtype):
                raise SpecificationError(f'dtype must be a torch.dtype, not {type(parsed_dtype)}')
            return parsed_dtype

        if shape is not None:
            cls.shape = shape_property
        if dtype is not None:
            cls.dtype = dtype_property

    def check_invariants(self) -> None:
        """Check that the dataclass invariants are satisfied."""
        memo = self._memo
        for elem in fields(self):
            name = elem.name
            expected_type = elem.type
            value = getattr(self, name)
            if get_origin(expected_type) is Annotated:
                expected_type, *annotations = get_args(expected_type)
            else:
                annotations = []
            if not isinstance(value, expected_type):
                raise TypeError(f'Expected {expected_type} for {name}, got {type(value)}')
            for annotation in annotations:
                # there could be other annotations not related to the shape and dtype
                if isinstance(annotation, Annotation):
                    try:
                        memo = annotation.check(value, memo=memo, strict=True)
                    except RuntimeCheckError as e:
                        raise type(e)(
                            f'Dataclass invariant violated for {self.__class__.__name__}.{name}: {e}'
                        ) from None
        object.__setattr__(self, '_memo', memo)  # works for frozen dataclasses
