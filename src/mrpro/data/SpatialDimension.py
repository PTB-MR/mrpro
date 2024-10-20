"""SpatialDimension dataclass."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast, overload, get_args

import numpy as np
import torch
from numpy.typing import ArrayLike

import mrpro.utils.typing as type_utils
from mrpro.data.MoveDataMixin import MoveDataMixin

VectorTypes = torch.Tensor | torch.nn.Parameter
ScalarTypes = int | float
T = TypeVar('T', torch.Tensor, int, float)
# Here we have to specify all subclasses of  torch.Tensor we support
# as it is not possible to write "either int, float or any subclass of torch.Tensor"i
T_co = TypeVar('T_co', torch.Tensor, torch.nn.Parameter, int, float, covariant=True)
T_co_float = TypeVar('T_co_float', float, torch.Tensor, torch.nn.Parameter, covariant=True)
T_co_vector = TypeVar('T_co_vector', torch.Tensor, torch.nn.Parameter, covariant=True)
T_co_scalar = TypeVar('T_co_scalar', int, float, covariant=True)


def _as_vectortype(x: ArrayLike) -> VectorTypes:
    """Convert ArrayLike to VectorType."""
    if isinstance(x, VectorTypes) and type(x) in get_args(VectorTypes):
        # exact type match
        return x
    if isinstance(x, VectorTypes):
        # subclass of torch.Tensor
        return torch.as_tensor(x)
    else:
        # any other ArrayLike (which is defined as convert to numpy array)
        return torch.as_tensor(np.asarray(x))


class XYZ(Protocol[T]):
    """Protocol for structures with attributes x, y and z of type T."""

    x: T
    y: T
    z: T


@dataclass(slots=True)
class SpatialDimension(MoveDataMixin, Generic[T_co]):
    """Spatial dataclass of float/int/tensors (z, y, x)."""

    z: T_co
    y: T_co
    x: T_co

    @classmethod
    def from_xyz(cls, data: XYZ[T_co]) -> SpatialDimension[T_co]:
        """Create a SpatialDimension from something with (.x .y .z) parameters.

        Parameters
        ----------
        data
            should implement .x .y .z. For example ismrmrd's matrixSizeType.
        """
        return cls(data.z, data.y, data.x)

    @staticmethod
    def from_array_xyz(
        data: ArrayLike,
    ) -> SpatialDimension[torch.Tensor]:
        """Create a SpatialDimension from an arraylike interface.

        Parameters
        ----------
        data
            shape (..., 3) in the order (x,y,z)
        """
        data_ = _as_vectortype(data)
        if np.size(data, -1) != 3:
            raise ValueError(f'Expected last dimension to be 3, got {np.size(data, -1)}')

        x = data_[..., 0]
        y = data_[..., 1]
        z = data_[..., 2]

        return SpatialDimension(z, y, x)

    @staticmethod
    def from_array_zyx(
        data: ArrayLike,
    ) -> SpatialDimension[torch.Tensor]:
        """Create a SpatialDimension from an arraylike interface.

        Parameters
        ----------
        data
            shape (..., 3) in the order (z,y,x)
        """
        data = _as_vectortype(data)
        if np.size(data, -1) != 3:
            raise ValueError(f'Expected last dimension to be 3, got {np.size(data, -1)}')

        x = data[..., 2]
        y = data[..., 1]
        z = data[..., 0]

        return SpatialDimension(z, y, x)

    @property
    def zyx(self) -> tuple[T_co, T_co, T_co]:
        """Return a z,y,x tuple."""
        return (self.z, self.y, self.x)

    def __str__(self) -> str:
        """Return a string representation of the SpatialDimension."""
        return f'z={self.z}, y={self.y}, x={self.x}'

    def __getitem__(
        self: SpatialDimension[T_co_vector], idx: type_utils.TorchIndexerType
    ) -> SpatialDimension[T_co_vector]:
        """Get SpatialDimension item."""
        if not all(isinstance(el, VectorTypes) for el in self.zyx):
            raise IndexError('Cannot index SpatialDimension with non-indexable members')
        return SpatialDimension(
            cast(T_co_vector, self.z[idx]), cast(T_co_vector, self.y[idx]), cast(T_co_vector, self.x[idx])
        )

    def __setitem__(self: SpatialDimension[T_co_vector], idx: type_utils.TorchIndexerType, other: SpatialDimension):
        """Set SpatialDimension item."""
        if not all(isinstance(el, VectorTypes) for el in self.zyx):
            raise IndexError('Cannot index SpatialDimension with non-indexable members')
        self.z[idx] = cast(T_co_vector, other.z)
        self.y[idx] = cast(T_co_vector, other.y)
        self.x[idx] = cast(T_co_vector, other.x)

    def apply_(self: SpatialDimension[T_co], func: Callable[[T_co], T_co] | None = None) -> SpatialDimension[T_co]:
        """Apply function to each of x,y,z in-place.

        Parameters
        ----------
        func
            function to apply to each of x,y,z
            None is interpreted as the identity function.
        """
        if func is not None:
            self.z = func(self.z)
            self.y = func(self.y)
            self.x = func(self.x)
        return self

    def apply(self: SpatialDimension[T_co], func: Callable[[T_co], T_co] | None = None) -> SpatialDimension[T_co]:
        """Apply function to each of x,y,z.

        Parameters
        ----------
        func
            function to apply to each of x,y,z
            None is interpreted as the identity function.
        """

        def func_(x: Any) -> T_co:  # noqa: ANN401
            if isinstance(x, torch.Tensor):
                # use clone for autograd
                x = x.clone()
            else:
                x = deepcopy(x)
            if func is None:
                return x
            else:
                return func(x)

        return self.__class__(func_(self.z), func_(self.y), func_(self.x))

    def clone(self: SpatialDimension[T_co]) -> SpatialDimension[T_co]:
        """Return a deep copy of the SpatialDimension."""
        return self.apply()

    @overload
    def __mul__(self: SpatialDimension[T_co], other: T_co | SpatialDimension[T_co]) -> SpatialDimension[T_co]: ...

    @overload
    def __mul__(self: SpatialDimension, other: SpatialDimension[T_co_vector]) -> SpatialDimension[T_co_vector]: ...

    @overload
    def __mul__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __mul__(
        self: SpatialDimension[T_co_float], other: float | SpatialDimension[float]
    ) -> SpatialDimension[T_co_float]: ...

    def __mul__(self: SpatialDimension[T_co], other: float | T_co | SpatialDimension) -> SpatialDimension:
        """Multiply SpatialDimension with numeric other or SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z * other.z, self.y * other.y, self.x * other.x)
        return SpatialDimension(self.z * other, self.y * other, self.x * other)

    # FIXME
    # The right-handed ops have a type ignore because of the type hinting in torch.Tensor being wrong.

    @overload
    def __rmul__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: SpatialDimension[int], other: float) -> SpatialDimension[float]: ...

    @overload
    def __rmul__(self: SpatialDimension[T_co_float], other: float) -> SpatialDimension[T_co_float]: ...

    def __rmul__(self: SpatialDimension[T_co], other: float | T_co | SpatialDimension) -> SpatialDimension:  # type: ignore[misc]
        """Right multiply SpatialDimension with numeric other or SpatialDimension."""
        return self.__mul__(other)

    @overload
    def __truediv__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __truediv__(self: SpatialDimension, other: SpatialDimension[T_co_vector]) -> SpatialDimension[T_co_vector]: ...

    @overload
    def __truediv__(self: SpatialDimension[T_co], other: T_co | SpatialDimension[T_co]) -> SpatialDimension[T_co]: ...

    @overload
    def __truediv__(
        self: SpatialDimension[T_co_float], other: float | SpatialDimension[float]
    ) -> SpatialDimension[T_co_float]: ...

    def __truediv__(self: SpatialDimension[T_co], other: float | T_co | SpatialDimension) -> SpatialDimension:
        """Divide SpatialDimension by numeric other or SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z / other.z, self.y / other.y, self.x / other.x)
        return SpatialDimension(self.z / other, self.y / other, self.x / other)

    @overload
    def __rtruediv__(self: SpatialDimension[int], other: float) -> SpatialDimension[float]: ...
    @overload
    def __rtruediv__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...  # type: ignore[misc]

    @overload
    def __rtruediv__(self: SpatialDimension[T_co_float], other: float) -> SpatialDimension[T_co_float]: ...

    def __rtruediv__(self: SpatialDimension[T_co], other: float | T_co) -> SpatialDimension:  # type: ignore[misc]
        """Divide SpatialDimension or numeric other by SpatialDimension."""
        return SpatialDimension(other / self.z, other / self.y, other / self.x)

    @overload
    def __add__(self: SpatialDimension[T_co], other: T_co | SpatialDimension[T_co]) -> SpatialDimension[T_co]: ...

    @overload
    def __add__(self: SpatialDimension, other: SpatialDimension[T_co_vector]) -> SpatialDimension[T_co_vector]: ...

    @overload
    def __add__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __add__(
        self: SpatialDimension[T_co_float], other: float | SpatialDimension[float]
    ) -> SpatialDimension[T_co_float]: ...

    def __add__(self: SpatialDimension[T_co], other: float | T_co | SpatialDimension) -> SpatialDimension:
        """Add SpatialDimension or numeric other to SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z + other.z, self.y + other.y, self.x + other.x)
        return SpatialDimension(self.z + other, self.y + other, self.x + other)

    @overload
    def __radd__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...  # type: ignore[misc]

    @overload
    def __radd__(self: SpatialDimension[int], other: float) -> SpatialDimension[float]: ...

    @overload
    def __radd__(self: SpatialDimension[T_co_float], other: float) -> SpatialDimension[T_co_float]: ...

    def __radd__(self: SpatialDimension[T_co], other: float | T_co) -> SpatialDimension:  # type: ignore[misc]
        """Right add numeric other to SpatialDimension."""
        return self.__add__(other)

    @overload
    def __floordiv__(self: SpatialDimension[T_co], other: T_co | SpatialDimension[T_co]) -> SpatialDimension[T_co]: ...

    @overload
    def __floordiv__(
        self: SpatialDimension[int], other: float | SpatialDimension[float]
    ) -> SpatialDimension[float]: ...

    @overload
    def __floordiv__(self: SpatialDimension, other: SpatialDimension[T_co_vector]) -> SpatialDimension[T_co_vector]: ...

    @overload
    def __floordiv__(
        self: SpatialDimension[T_co_float], other: float | SpatialDimension[float]
    ) -> SpatialDimension[T_co_float]: ...

    def __floordiv__(self: SpatialDimension[T_co], other: float | T_co | SpatialDimension) -> SpatialDimension:
        """Floor divide SpatialDimension by numeric other."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z // other.z, self.y // other.y, self.x // other.x)
        return SpatialDimension(self.z // other, self.y // other, self.x // other)

    @overload
    def __rfloordiv__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...  # type: ignore[misc]

    @overload
    def __rfloordiv__(self: SpatialDimension[int], other: float) -> SpatialDimension[float]: ...

    @overload
    def __rfloordiv__(self: SpatialDimension[T_co_float], other: float) -> SpatialDimension[T_co_float]: ...

    def __rfloordiv__(self: SpatialDimension[T_co], other: float | T_co) -> SpatialDimension:  # type: ignore[misc]
        """Floor divide other by SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(other.z // self.z, other.y // self.y, other.x // self.x)
        return SpatialDimension(other // self.z, other // self.y, other // self.x)

    @overload
    def __sub__(self: SpatialDimension[T_co], other: T_co | SpatialDimension[T_co]) -> SpatialDimension[T_co]: ...

    @overload
    def __sub__(self: SpatialDimension, other: SpatialDimension[T_co_vector]) -> SpatialDimension[T_co_vector]: ...

    @overload
    def __sub__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __sub__(
        self: SpatialDimension[T_co_float], other: float | SpatialDimension[float]
    ) -> SpatialDimension[T_co_float]: ...

    def __sub__(self: SpatialDimension[T_co], other: float | T_co | SpatialDimension) -> SpatialDimension:
        """Subtract SpatialDimension or numeric other to SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z - other.z, self.y - other.y, self.x - other.x)
        return SpatialDimension(self.z - other, self.y - other, self.x - other)

    @overload
    def __rsub__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...  # type: ignore[misc]

    @overload
    def __rsub__(self: SpatialDimension[int], other: float) -> SpatialDimension[float]: ...

    @overload
    def __rsub__(self: SpatialDimension[T_co_float], other: float) -> SpatialDimension[T_co_float]: ...

    def __rsub__(self: SpatialDimension[T_co], other: float | T_co) -> SpatialDimension:  # type: ignore[misc]
        """Subtract SpatialDimension from numeric other or SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(other.z - self.z, other.y - self.y, other.x - self.x)
        return SpatialDimension(other - self.z, other - self.y, other - self.x)

    def __neg__(self: SpatialDimension[T_co]) -> SpatialDimension[T_co]:
        """Negate SpatialDimension."""
        return SpatialDimension(-self.z, -self.y, -self.x)

    @overload
    def __eq__(self: SpatialDimension[T_co_scalar], other: object) -> bool: ...
    @overload
    def __eq__(self: SpatialDimension[T_co_vector], other: SpatialDimension[T_co_vector]) -> T_co_vector: ...

    def __eq__(
        self: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
        other: object | SpatialDimension[T_co_vector],
    ) -> bool | T_co_vector:
        """Check if self is equal to other."""
        if not isinstance(other, SpatialDimension):
            return NotImplemented
        return (self.z == other.z) & (self.y == other.y) & (self.x == other.x)

    @overload
    def __lt__(self: SpatialDimension[T_co_vector], other: SpatialDimension[T_co_vector]) -> T_co_vector: ...
    @overload
    def __lt__(self: SpatialDimension[T_co_scalar], other: SpatialDimension[T_co_scalar]) -> bool: ...
    def __lt__(
        self: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
        other: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
    ) -> bool | T_co_vector:
        """Check if self is less than other."""
        if not isinstance(other, SpatialDimension):
            return NotImplemented
        return (self.x < other.x) & (self.y < other.y) & (self.z < other.z)

    @overload
    def __le__(self: SpatialDimension[T_co_vector], other: SpatialDimension[T_co_vector]) -> T_co_vector: ...
    @overload
    def __le__(self: SpatialDimension[T_co_scalar], other: SpatialDimension[T_co_scalar]) -> bool: ...
    def __le__(
        self: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
        other: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
    ) -> bool | T_co_vector:
        """Check if self is less than or equal to other."""
        if not isinstance(other, SpatialDimension):
            return NotImplemented
        return (self.x <= other.x) & (self.y <= other.y) & (self.z <= other.z)

    @overload
    def __gt__(self: SpatialDimension[T_co_vector], other: SpatialDimension[T_co_vector]) -> T_co_vector: ...
    @overload
    def __gt__(self: SpatialDimension[T_co_scalar], other: SpatialDimension[T_co_scalar]) -> bool: ...
    def __gt__(
        self: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
        other: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
    ) -> bool | T_co_vector:
        """Check if self is greater than other."""
        if not isinstance(other, SpatialDimension):
            return NotImplemented
        return (self.x > other.x) & (self.y > other.y) & (self.z > other.z)

    @overload
    def __ge__(self: SpatialDimension[T_co_vector], other: SpatialDimension[T_co_vector]) -> T_co_vector: ...
    @overload
    def __ge__(self: SpatialDimension[T_co_scalar], other: SpatialDimension[T_co_scalar]) -> bool: ...
    def __ge__(
        self: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
        other: SpatialDimension[T_co_scalar] | SpatialDimension[T_co_vector],
    ) -> bool | T_co_vector:
        """Check if self is greater than or equal to other."""
        if not isinstance(other, SpatialDimension):
            return NotImplemented
        return (self.x >= other.x) & (self.y >= other.y) & (self.z >= other.z)

    def __post_init__(self):
        """Ensure that the data is of matching shape."""
        if not all(isinstance(val, (int | float)) for val in self.zyx):
            try:
                zyx = [_as_vectortype(v) for v in self.zyx]
                self.z, self.y, self.x = torch.broadcast_tensors(*zyx)
            except RuntimeError:
                raise ValueError('The shapes of the tensors do not match') from None

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the x, y, and z.

        Returns
        -------
            Empty tuple if x, y, and z are scalar types, otherwise shape

        Raises
        ------
            ValueError if the shapes are not equal
        """
        if isinstance(self.x, ScalarTypes) and isinstance(self.y, ScalarTypes) and isinstance(self.z, ScalarTypes):
            return ()
        elif (
            isinstance(self.x, VectorTypes)
            and isinstance(self.y, VectorTypes)
            and isinstance(self.z, VectorTypes)
            and self.x.shape == self.y.shape == self.z.shape
        ):
            return self.x.shape
        else:
            raise ValueError('Inconsistent shapes')
