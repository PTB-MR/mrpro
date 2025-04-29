"""Base class for all data classes."""

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from copy import copy as shallowcopy
from copy import deepcopy
from typing import ClassVar, TypeAlias, cast

import torch
from typing_extensions import Any, Protocol, Self, TypeVar, dataclass_transform, overload, runtime_checkable

from mrpro.utils.indexing import HasIndex, Indexer
from mrpro.utils.reduce_repeat import reduce_repeat
from mrpro.utils.summarize import summarize_object
from mrpro.utils.typing import TorchIndexerType


@runtime_checkable
class HasReduceRepeats(Protocol):
    """Objects that have a _reduce_repeats method."""

    def _reduce_repeats_(self, tol: float = 1e-6, dim: Sequence[int] | None = None, recurse: bool = True) -> Self: ...


class InconsistentDeviceError(ValueError):
    """Raised if the devices of different fields differ.

    There is no single device that all fields are on, thus
    the overall device of the object cannot be determined.
    """

    def __init__(self, *devices):
        """Initialize.

        Parameters
        ----------
        devices
            The devices of the fields that differ.
        """
        super().__init__(f'Inconsistent devices found, found at least {", ".join(str(d) for d in devices)}')


class InconsistentShapeError(RuntimeError):
    """Raised if fields are not broadastable.

    The fields cannot be broadasted to a common shape.
    """

    def __init__(self, *shapes):
        """Initialize.

        Parameters
        ----------
        shapes
            The shapes of the fields.
        """
        super().__init__(f'The shapes of the fields are not broadcastable. Found shapes: {shapes}.')


T = TypeVar('T')


@dataclass_transform()
class Dataclass:
    """A supercharged dataclass with additional functionality.

    This class extends the functionality of the standard `dataclasses.dataclass` by adding
    - a `apply` method to apply a function to all fields
    - a `~Dataclass.clone` method to create a deep copy of the object
    - `~Dataclass.to`, `~Dataclass.cpu`, `~Dataclass.cuda` methods to move all tensor fields to a device

    It is intended to be used as a base class for all dataclasses in the `mrpro` package.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]
    __auto_reduce_repeats: bool
    __initialized: bool

    def __init_subclass__(  # noqa: D417
        cls,
        no_new_attributes: bool = True,
        auto_reduce_repeats: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Create a new dataclass subclass.

        Parameters
        ----------
        no_new_attributes
            If `True`, new attributes cannot be added to the class after it is created.
        auto_reduce_repeats
            If `True`, try to reduce dimensions only containing repeats to singleton.
            This will be done after init and post_init.
        """
        dataclasses.dataclass(cls, repr=False)  # type: ignore[call-overload]
        super().__init_subclass__(**kwargs)
        child_post_init = vars(cls).get('__post_init__')

        if no_new_attributes:

            def new_setattr(self: Dataclass, name: str, value: Any) -> None:  # noqa: ANN401
                """Set an attribute."""
                if not hasattr(self, name) and hasattr(self, '_Dataclass__initialized'):
                    raise AttributeError(f'Cannot set attribute {name} on {self.__class__.__name__}')
                object.__setattr__(self, name, value)

            cls.__setattr__ = new_setattr  # type: ignore[method-assign, assignment]

        cls.__auto_reduce_repeats = auto_reduce_repeats

        if child_post_init and child_post_init is not Dataclass.__post_init__:

            def chained_post_init(self: Dataclass, *args, **kwargs) -> None:
                child_post_init(self, *args, **kwargs)
                Dataclass.__post_init__(self)

            cls.__post_init__ = chained_post_init  # type: ignore[method-assign]

    def __post_init__(self) -> None:
        """Can be overridden in subclasses to add custom initialization logic."""
        self.__initialized = True
        if self.__auto_reduce_repeats:
            self._reduce_repeats_(recurse=False)

    def _reduce_repeats_(self, tol: float = 1e-6, dim: Sequence[int] | None = None, recurse: bool = True) -> Self:
        """Reduce repeated dimensions in fields to singleton.

        Parameters
        ----------
        tol
            tolerance.
        dim
            dimensions to try to reduce to singletons. `None` means all.
        recurse
            recurse into dataclass fields.
        """

        def apply_reduce(data: T) -> T:
            if isinstance(data, torch.Tensor):
                return cast(T, reduce_repeat(data, tol, dim))
            if isinstance(data, HasReduceRepeats) and not isinstance(data, Dataclass):
                data._reduce_repeats_(tol, dim)
            return cast(T, data)

        return self.apply_(apply_reduce, recurse=recurse)

    def items(self) -> Iterator[tuple[str, Any]]:
        """Get an iterator over names and values of fields."""
        for field in dataclasses.fields(self):
            name = field.name
            data = getattr(self, name)
            yield name, data

    def apply_(
        self: Self,
        function: Callable[[Any], Any] | None = None,
        *,
        memo: dict[int, Any] | None = None,
        recurse: bool = True,
    ) -> Self:
        """Apply a function to all children in-place.

        Parameters
        ----------
        function
            The function to apply to all fields. `None` is interpreted as a no-op.
        memo
            A dictionary to keep track of objects that the function has already been applied to,
            to avoid multiple applications. This is useful if the object has a circular reference.
        recurse
            If `True`, the function will be applied to all children that are `Dataclass` instances.
        """
        applied: Any

        if memo is None:
            memo = {}

        if function is None:
            return self

        for name, data in self.items():
            if id(data) in memo:
                # this works even if self is frozen
                object.__setattr__(self, name, memo[id(data)])
                continue
            if recurse and isinstance(data, Dataclass):
                applied = data.apply_(function, memo=memo)
            else:
                applied = function(data)
            memo[id(data)] = applied
            object.__setattr__(self, name, applied)
        return self

    def apply(
        self: Self,
        function: Callable[[Any], Any] | None = None,
        *,
        recurse: bool = True,
    ) -> Self:
        """Apply a function to all children. Returns a new object.

        Parameters
        ----------
        function
            The function to apply to all fields. `None` is interpreted as a no-op.
        recurse
            If `True`, the function will be applied to all children that are `Dataclass` instances.
        """
        new = self.clone().apply_(function, recurse=recurse)
        return new

    # region Move to device/dtype
    @overload
    def to(
        self,
        device: str | torch.device | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        *,
        copy: bool = False,
        memory_format: torch.memory_format | None = None,
    ) -> Self: ...

    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: bool = False,
        *,
        copy: bool = False,
        memory_format: torch.memory_format | None = None,
    ) -> Self: ...

    @overload
    def to(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = False,
        *,
        copy: bool = False,
        memory_format: torch.memory_format | None = None,
    ) -> Self: ...

    def to(self, *args, **kwargs) -> Self:
        """Perform dtype and/or device conversion of data.

        A `torch.dtype` and `torch.device` are inferred from the arguments
        args and kwargs. Please have a look at the
        documentation of `torch.Tensor.to` for more details.

        A new instance of the dataclass will be returned.

        The conversion will be applied to all Tensor- or Module
        fields of the dataclass, and to all fields that implement
        the `Dataclass`.

        The dtype-type, i.e. float or complex will always be preserved,
        but the precision of floating point dtypes might be changed.

        Example:
        If called with ``dtype=torch.float32`` OR ``dtype=torch.complex64``:

        - A ``complex128`` tensor will be converted to ``complex64``
        - A ``float64`` tensor will be converted to ``float32``
        - A ``bool`` tensor will remain ``bool``
        - An ``int64`` tensor will remain ``int64``

        If other conversions are desired, please use the `~torch.Tensor.to` method of
        the fields directly.

        If the copy argument is set to `True` (default), a deep copy will be returned
        even if no conversion is necessary.
        If two fields are views of the same data before, in the result they will be independent
        copies if copy is set to `True` or a conversion is necessary.
        If set to `False`, some Tensors might be shared between the original and the new object.
        """
        # Parse the arguments of the three overloads and call _to with the parsed arguments
        parsedType: TypeAlias = tuple[
            str | torch.device | int | None, torch.dtype | None, bool, bool, torch.memory_format
        ]

        def parse3(
            other: torch.Tensor,
            non_blocking: bool = False,
            copy: bool = False,
        ) -> parsedType:
            return other.device, other.dtype, non_blocking, copy, torch.preserve_format

        def parse2(
            dtype: torch.dtype,
            non_blocking: bool = False,
            copy: bool = False,
            memory_format: torch.memory_format = torch.preserve_format,
        ) -> parsedType:
            return None, dtype, non_blocking, copy, memory_format

        def parse1(
            device: str | torch.device | int | None = None,
            dtype: None | torch.dtype = None,
            non_blocking: bool = False,
            copy: bool = False,
            memory_format: torch.memory_format = torch.preserve_format,
        ) -> parsedType:
            return device, dtype, non_blocking, copy, memory_format

        if (args and isinstance(args[0], torch.Tensor)) or 'tensor' in kwargs:
            # overload 3 ("tensor" specifies the dtype and device)
            device, dtype, non_blocking, copy, memory_format = parse3(*args, **kwargs)
        elif args and isinstance(args[0], torch.dtype):
            # overload 2 (no device specified, only dtype)
            device, dtype, non_blocking, copy, memory_format = parse2(*args, **kwargs)
        else:
            # overload 1 (device and dtype specified)
            device, dtype, non_blocking, copy, memory_format = parse1(*args, **kwargs)
        return self._to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format, copy=copy)

    def _to(
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
        shared_memory: bool = False,
        copy: bool = False,
        memo: dict | None = None,
    ) -> Self:
        """Move data to device and convert dtype if necessary.

        This method is called by `.to()`, `.cuda()`, `.cpu()`,
        `.double()`, and so on. It should not be called directly.

        See `Dataclass.to()` for more details.

        Parameters
        ----------
        device
            The destination device.
        dtype
            The destination dtype.
        non_blocking
            If `True` and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        memory_format
            The desired memory format of returned tensor.
        shared_memory
            If `True` and the target device is CPU, the tensors will reside in shared memory.
            Otherwise, the argument has no effect.
        copy
            If `True`, the returned tensor will always be a copy, even if the input was already on the correct device.
            This will also create new tensors for views.
        memo
            A dictionary to keep track of already converted objects to avoid multiple conversions.
        """
        new = shallowcopy(self) if copy or not isinstance(self, torch.nn.Module) else self

        if memo is None:
            memo = {}

        def _tensor_to(data: torch.Tensor) -> torch.Tensor:
            """Move tensor to device and convert dtype if necessary."""
            new_dtype: torch.dtype | None
            if dtype is not None and data.dtype.is_floating_point:
                new_dtype = dtype.to_real()
            elif dtype is not None and data.dtype.is_complex:
                new_dtype = dtype.to_complex()
            else:
                # bool or int: keep as is
                new_dtype = None
            data = data.to(
                device,
                new_dtype,
                non_blocking=non_blocking,
                memory_format=memory_format,
                copy=copy,
            )
            if shared_memory:
                data.share_memory_()
            return data

        def _module_to(data: torch.nn.Module) -> torch.nn.Module:
            if copy:
                data = deepcopy(data)
            return data._apply(_tensor_to, recurse=True)

        def _mixin_to(obj: Dataclass) -> Dataclass:
            return obj._to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                memory_format=memory_format,
                shared_memory=shared_memory,
                copy=copy,
                memo=memo,
            )

        def _convert(data: T) -> T:
            converted: Any  # https://github.com/python/mypy/issues/10817
            if isinstance(data, torch.Tensor):
                converted = _tensor_to(data)
            elif isinstance(data, Dataclass):
                converted = _mixin_to(data)
            elif isinstance(data, torch.nn.Module):
                converted = _module_to(data)
            else:
                converted = data
            return cast(T, converted)

        # manual recursion allows us to do the copy only once
        new.apply_(_convert, memo=memo, recurse=False)
        return new

    def cuda(
        self,
        device: torch.device | str | int | None = None,
        *,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
        copy: bool = False,
    ) -> Self:
        """Put object in CUDA memory.

        Parameters
        ----------
        device
            The destination GPU device. Defaults to the current CUDA device.
        non_blocking
            If `True` and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        memory_format
            The desired memory format of returned tensor.
        copy:
            If `True`, the returned tensor will always be a copy, even if the input was already on the correct device.
            This will also create new tensors for views.
        """
        if device is None:
            device = torch.device(torch.cuda.current_device())
        return self._to(device=device, dtype=None, memory_format=memory_format, non_blocking=non_blocking, copy=copy)

    def cpu(self, *, memory_format: torch.memory_format = torch.preserve_format, copy: bool = False) -> Self:
        """Put in CPU memory.

        Parameters
        ----------
        memory_format
            The desired memory format of returned tensor.
        copy
            If `True`, the returned tensor will always be a copy, even if the input was already on the correct device.
            This will also create new tensors for views.
        """
        return self._to(device='cpu', dtype=None, non_blocking=True, memory_format=memory_format, copy=copy)

    def double(self, *, memory_format: torch.memory_format = torch.preserve_format, copy: bool = False) -> Self:
        """Convert all float tensors to double precision.

        converts ``float`` to ``float64`` and ``complex`` to ``complex128``


        Parameters
        ----------
        memory_format
            The desired memory format of returned tensor.
        copy
            If `True`, the returned tensor will always be a copy, even if the input was already on the correct device.
            This will also create new tensors for views.
        """
        return self._to(dtype=torch.float64, memory_format=memory_format, copy=copy)

    def half(self, *, memory_format: torch.memory_format = torch.preserve_format, copy: bool = False) -> Self:
        """Convert all float tensors to half precision.

        converts ``float`` to ``float16`` and ``complex`` to ``complex32``


        Parameters
        ----------
        memory_format
            The desired memory format of returned tensor.
        copy
            If `True`, the returned tensor will always be a copy, even if the input was already on the correct device.
            This will also create new tensors for views.
        """
        return self._to(dtype=torch.float16, memory_format=memory_format, copy=copy)

    def single(self, *, memory_format: torch.memory_format = torch.preserve_format, copy: bool = False) -> Self:
        """Convert all float tensors to single precision.

        converts ``float`` to ``float32`` and ``complex`` to ``complex64``


        Parameters
        ----------
        memory_format
            The desired memory format of returned tensor.
        copy
            If `True`, the returned tensor will always be a copy, even if the input was already on the correct device.
            This will also create new tensors for views.
        """
        return self._to(dtype=torch.float32, memory_format=memory_format, copy=copy)

    # endregion Move to device/dtype

    # region Properties
    @property
    def device(self) -> torch.device | None:
        """Return the device of the tensors.

        Looks at each field of a dataclass implementing a device attribute,
        such as `torch.Tensor` or `Dataclass` instances. If the devices
        of the fields differ, an :py:exc:`~mrpro.data.InconsistentDeviceError` is raised, otherwise
        the device is returned. If no field implements a device attribute,
        None is returned.

        Raises
        ------
        :py:exc:`InconsistentDeviceError`
            If the devices of different fields differ.

        Returns
        -------
            The device of the fields or `None` if no field implements a `device` attribute.
        """
        device: None | torch.device = None
        for _, data in self.items():
            if not hasattr(data, 'device'):
                continue
            current_device = getattr(data, 'device', None)
            if current_device is None:
                continue
            if device is None:
                device = current_device
            elif device != current_device:
                raise InconsistentDeviceError(current_device, device)
        return device

    def clone(self: Self) -> Self:
        """Return a deep copy of the object."""
        return self._to(device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format, copy=True)

    @property
    def is_cuda(self) -> bool:
        """Return `True` if all tensors are on a single CUDA device.

        Checks all tensor attributes of the dataclass for their device,
        (recursively if an attribute is a `Dataclass`)


        Returns `False` if not all tensors are on the same CUDA devices, or if the device is inconsistent,
        returns `True` if the data class has no tensors as attributes.
        """
        try:
            device = self.device
        except InconsistentDeviceError:
            return False
        if device is None:
            return True
        return device.type == 'cuda'

    @property
    def is_cpu(self) -> bool:
        """Return True if all tensors are on the CPU.

        Checks all tensor attributes of the dataclass for their device,
        (recursively if an attribute is a `Dataclass`)

        Returns `False` if not all tensors are on cpu or if the device is inconsistent,
        returns `True` if the data class has no tensors as attributes.
        """
        try:
            device = self.device
        except InconsistentDeviceError:
            return False
        if device is None:
            return True
        return device.type == 'cpu'

    @property
    def shape(self) -> torch.Size:
        """Return the broadcasted shape of all tensor/data fields.

        Each field of this dataclass is broadcastable to this shape.

        Returns
        -------
            The broadcasted shape of all fields.

        Raises
        ------
        :py:exc:`InconsistentShapeError`
            If the shapes cannot be broadcasted.
        """
        shapes = []
        for _, data in self.items():
            if not hasattr(data, 'shape'):
                continue
            shapes.append(data.shape)
        try:
            return torch.broadcast_shapes(*shapes)
        except RuntimeError:
            raise InconsistentShapeError(*shapes) from None

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the dataclass.

        This is the number of dimensions of the broadcasted shape of all fields.
        """
        return len(self.shape)

    # endregion Properties

    # region Representation
    def __repr__(self) -> str:
        """Get string representation of Dataclass."""
        header = [type(self).__name__]

        try:
            device = self.device
            if device:
                header.append(f'on device "{device}"')
        except RuntimeError:
            header.append('on mixed devices')

        try:
            if shape := self.shape:
                header.append(f'with (broadcasted) shape {list(shape)!s}')
        except RuntimeError:
            header.append('with inconsistent shape')

        output = ' '.join(header) + '.\n'

        output += '\n'.join(
            f'  {field.name}: {summarize_object(value)}'
            for field in dataclasses.fields(self)
            if not (field.name.startswith('_') or (value := getattr(self, field.name, None)) is None)
        )
        return output

    # We return the same for __repr__ and __str__.
    # This break the "_str_ if for users, _repr_ for developers rule" of python.
    # But it makes interactive work on repl or notebooks easier, as `obj` can be used instead
    # of `print(obj)`. It would be infeasable for most dataclasses to implement a proper  __repr__
    # that uniquely describes the data and could be used to recreate the object anyways.

    def __str__(self) -> str:
        """Return the same as __repr__."""
        return repr(self)

    def __shortstr__(self) -> str:
        """Return a short string representation."""
        output = type(self).__name__
        if self.shape:
            output = output + f'<{", ".join(map(str, self.shape))}>'
        return output

    # endregion Representation

    # region Indexing
    def __getitem__(self, index: TorchIndexerType | Indexer) -> Self:
        """Index the dataclass."""
        indexer = index if isinstance(index, Indexer) else Indexer(self.shape, index)
        memo: dict[int, Any] = {}

        def apply_index(data: T) -> T:
            if isinstance(data, torch.Tensor):
                return cast(T, indexer(data))
            if isinstance(data, Dataclass):
                indexed = shallowcopy(data)
                indexed.apply_(apply_index, memo=memo, recurse=False)
                return cast(T, indexed)
            if isinstance(data, HasIndex):
                # Rotation
                return cast(T, data._index(indexer))
            return cast(T, data)

        new = shallowcopy(self)
        new.apply_(apply_index, memo=memo, recurse=False)
        return new

    # endregion Indexing
