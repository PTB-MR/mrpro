"""SpatialDimension dataclass."""

from __future__ import annotations

import operator
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, overload

import numpy as np
import torch
from numpy.typing import ArrayLike

import mrpro.utils.typing as typeing_utils
from mrpro.data.MoveDataMixin import MoveDataMixin

T = TypeVar('T', int, float, np.ndarray, torch.Tensor)
T_co = TypeVar('T_co', int, float, np.ndarray, torch.Tensor, covariant=True)
T_co_float = TypeVar('T_co_float', float, np.ndarray, torch.Tensor, covariant=True)
T_co_vector = TypeVar('T_co_vector', np.ndarray, torch.Tensor, covariant=True)


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
        if not isinstance(data, np.ndarray | torch.Tensor):
            data = np.asarray(data)

        if np.size(data, -1) != 3:
            raise ValueError(f'Expected last dimension to be 3, got {np.size(data, -1)}')

        x = torch.as_tensor(data[..., 0])
        y = torch.as_tensor(data[..., 1])
        z = torch.as_tensor(data[..., 2])

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
        data = torch.flip(torch.as_tensor(data), (-1,))
        return SpatialDimension.from_array_xyz(data)

    @property
    def zyx(self) -> tuple[T_co, T_co, T_co]:
        """Return a z,y,x tuple."""
        return (self.z, self.y, self.x)

    def __str__(self) -> str:
        """Return a string representation of the SpatialDimension."""
        return f'z={self.z}, y={self.y}, x={self.x}'

    def __getitem__(
        self: SpatialDimension[torch.Tensor], idx: typeing_utils.IndexerType
    ) -> SpatialDimension[torch.Tensor]:
        """Get SpatialDimension item."""
        if not all(isinstance(el, torch.Tensor) for el in self.zyx):
            raise IndexError('Cannot index SpatialDimension with non-tensor')
        return SpatialDimension(self.z[idx], self.y[idx], self.x[idx])

    def __setitem__(self: SpatialDimension[torch.Tensor], idx: typeing_utils.IndexerType, other: SpatialDimension):
        """Set SpatialDimension item."""
        if not all(isinstance(el, torch.Tensor) for el in self.zyx):
            raise IndexError('Cannot index SpatialDimension with non-tensor')
        self.z[idx] = other.z
        self.y[idx] = other.y
        self.x[idx] = other.x

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
    def __rmul__(self: SpatialDimension[torch.Tensor], other: torch.Tensor) -> SpatialDimension[torch.Tensor]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...
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
    def __rtruediv__(self: SpatialDimension[torch.Tensor], other: torch.Tensor) -> SpatialDimension[torch.Tensor]: ...  # type: ignore[misc]
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
    def __radd__(self: SpatialDimension[torch.Tensor], other: torch.Tensor) -> SpatialDimension[torch.Tensor]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...

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
    def __rfloordiv__(self: SpatialDimension[T_co], other: T_co) -> SpatialDimension[T_co]: ...

    @overload
    def __rfloordiv__(self: SpatialDimension[int], other: float) -> SpatialDimension[float]: ...

    @overload
    def __rfloordiv__(self: SpatialDimension[T_co_float], other: float) -> SpatialDimension[T_co_float]: ...

    def __rfloordiv__(self: SpatialDimension[T_co], other: float | T_co) -> SpatialDimension:
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
    def __rsub__(self: SpatialDimension[torch.Tensor], other: torch.Tensor) -> SpatialDimension[torch.Tensor]: ...  # type: ignore[misc]

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

    def __compare__(
        self: SpatialDimension[T_co],
        other: SpatialDimension[T_co],
        op: Callable[[T_co, T_co], bool | torch.Tensor | np.ndarray],
    ) -> bool:
        """Perform comparison operation on SpatialDimension."""
        x = op(self.x, other.x)
        y = op(self.y, other.y)
        z = op(self.z, other.z)
        xx = x if isinstance(x, bool) else bool(x.all())
        yy = y if isinstance(y, bool) else bool(y.all())
        zz = z if isinstance(z, bool) else bool(z.all())
        return xx and yy and zz

    def __eq__(self: SpatialDimension[T_co], other: object) -> bool:
        """Check if self is equal to other."""
        if not isinstance(other, SpatialDimension):
            return NotImplemented
        else:
            return self.__compare__(other, operator.eq)

    def __lt__(self: SpatialDimension[T_co], other: SpatialDimension[T_co]) -> bool:
        """Check if self is less than other."""
        return self.__compare__(other, operator.lt)

    def __le__(self: SpatialDimension[T_co], other: SpatialDimension[T_co]) -> bool:
        """Check if sel is less of equal than other."""
        return self.__compare__(other, operator.le)

    def __gt__(self: SpatialDimension[T_co], other: SpatialDimension[T_co]) -> bool:
        """Check if self is greater than other."""
        return self.__compare__(other, operator.gt)

    def __ge__(self: SpatialDimension[T_co], other: SpatialDimension[T_co]) -> bool:
        """Check if self is greater or equal than other."""
        return self.__compare__(other, operator.ge)
