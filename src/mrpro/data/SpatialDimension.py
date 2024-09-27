"""SpatialDimension dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, overload

import numpy as np
import torch
from numpy.typing import ArrayLike

import mrpro.utils.typing as typeing_utils
from mrpro.data.MoveDataMixin import MoveDataMixin

T = TypeVar('T', int, float, torch.Tensor)


class XYZ(Protocol[T]):
    """Protocol for structures with attributes x, y and z of type T."""

    x: T
    y: T
    z: T


@dataclass(slots=True)
class SpatialDimension(MoveDataMixin, Generic[T]):
    """Spatial dataclass of float/int/tensors (z, y, x)."""

    z: T
    y: T
    x: T

    @classmethod
    def from_xyz(cls, data: XYZ[T], conversion: Callable[[T], T] | None = None) -> SpatialDimension[T]:
        """Create a SpatialDimension from something with (.x .y .z) parameters.

        Parameters
        ----------
        data
            should implement .x .y .z. For example ismrmrd's matrixSizeType.
        conversion,  optional
            will be called for each other to convert it
        """
        if conversion is not None:
            return cls(conversion(data.z), conversion(data.y), conversion(data.x))
        return cls(data.z, data.y, data.x)

    @staticmethod
    def from_array_xyz(
        data: ArrayLike,
        conversion: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> SpatialDimension[torch.Tensor]:
        """Create a SpatialDimension from an arraylike interface.

        Parameters
        ----------
        data
            shape (..., 3) in the order (x,y,z)
        conversion
            will be called for each other to convert it
        """
        if not isinstance(data, np.ndarray | torch.Tensor):
            data = np.asarray(data)

        if np.size(data, -1) != 3:
            raise ValueError(f'Expected last dimension to be 3, got {np.size(data, -1)}')

        x = torch.as_tensor(data[..., 0])
        y = torch.as_tensor(data[..., 1])
        z = torch.as_tensor(data[..., 2])

        if conversion is not None:
            x = conversion(x)
            y = conversion(y)
            z = conversion(z)
        return SpatialDimension(z, y, x)

    @staticmethod
    def from_array_zyx(
        data: ArrayLike,
        conversion: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> SpatialDimension[torch.Tensor]:
        """Create a SpatialDimension from an arraylike interface.

        Parameters
        ----------
        data
            shape (..., 3) in the order (z,y,x)
        conversion
            will be called for each other to convert it
        """
        data = torch.flip(torch.as_tensor(data), (-1,))
        return SpatialDimension.from_array_xyz(data, conversion)

    @property
    def zyx(self) -> tuple[T, T, T]:
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

    @overload
    def __mul__(self: SpatialDimension[int], other: int | SpatialDimension[int]) -> SpatialDimension[int]: ...
    @overload
    def __mul__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...
    @overload
    def __mul__(self: SpatialDimension[float], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __mul__(
        self: SpatialDimension[torch.Tensor], other: float | T | SpatialDimension
    ) -> SpatialDimension[torch.Tensor]: ...

    def __mul__(self: SpatialDimension[T], other: T | float | SpatialDimension) -> SpatialDimension:
        """Multiply SpatialDimension with numeric other or SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z * other.z, self.y * other.y, self.x * other.x)
        return SpatialDimension(self.z * other, self.y * other, self.x * other)

    @overload
    def __rmul__(self: SpatialDimension[int], other: int | SpatialDimension[int]) -> SpatialDimension[int]: ...
    @overload
    def __rmul__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...
    @overload
    def __rmul__(self: SpatialDimension[float], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __rmul__(  # type: ignore[misc]
        self: SpatialDimension[torch.Tensor], other: float | T | SpatialDimension
    ) -> SpatialDimension[torch.Tensor]: ...

    def __rmul__(self: SpatialDimension[T], other: T | float | SpatialDimension) -> SpatialDimension:  # type: ignore[misc]
        """Right multiply SpatialDimension with numeric other or SpatialDimension."""
        return self.__mul__(other)

    @overload
    def __truediv__(
        self: SpatialDimension[float], other: float | SpatialDimension[float]
    ) -> SpatialDimension[float]: ...

    @overload
    def __truediv__(
        self: SpatialDimension[torch.Tensor], other: float | T | SpatialDimension
    ) -> SpatialDimension[torch.Tensor]: ...

    def __truediv__(self: SpatialDimension[T], other: T | float | SpatialDimension) -> SpatialDimension:
        """Divide SpatialDimension with numeric other or SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z / other.z, self.y / other.y, self.x / other.x)
        return SpatialDimension(self.z / other, self.y / other, self.x / other)

    @overload
    def __rtruediv__(
        self: SpatialDimension[float], other: float | SpatialDimension[float]
    ) -> SpatialDimension[float]: ...

    @overload
    def __rtruediv__(  # type: ignore[misc]
        self: SpatialDimension[torch.Tensor], other: float | T | SpatialDimension
    ) -> SpatialDimension[torch.Tensor]: ...

    def __rtruediv__(self: SpatialDimension[T], other: T | float) -> SpatialDimension:  # type: ignore[misc]
        """Right divide SpatialDimension with numeric other."""
        return SpatialDimension(other / self.z, other / self.y, other / self.x)

    @overload
    def __add__(self: SpatialDimension[int], other: int | SpatialDimension[int]) -> SpatialDimension[int]: ...
    @overload
    def __add__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...
    @overload
    def __add__(self: SpatialDimension[float], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __add__(
        self: SpatialDimension[torch.Tensor], other: float | T | SpatialDimension
    ) -> SpatialDimension[torch.Tensor]: ...

    def __add__(self: SpatialDimension[T], other: T | float | SpatialDimension) -> SpatialDimension:
        """Add SpatialDimension or numeric other to SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z + other.z, self.y + other.y, self.x + other.x)
        return SpatialDimension(self.z + other, self.y + other, self.x + other)

    @overload
    def __radd__(self: SpatialDimension[int], other: int | SpatialDimension[int]) -> SpatialDimension[int]: ...
    @overload
    def __radd__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...
    @overload
    def __radd__(self: SpatialDimension[float], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    @overload
    def __radd__(  # type: ignore[misc]
        self: SpatialDimension[torch.Tensor], other: float | T | SpatialDimension
    ) -> SpatialDimension[torch.Tensor]: ...

    def __radd__(self: SpatialDimension[T], other: T | float | SpatialDimension) -> SpatialDimension:  # type: ignore[misc]
        """Right add numeric other to SpatialDimension."""
        return self.__add__(other)

    @overload
    def __sub__(self: SpatialDimension[int], other: int | SpatialDimension[int]) -> SpatialDimension[int]: ...
    @overload
    def __sub__(self: SpatialDimension[int], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...
    @overload
    def __sub__(self: SpatialDimension[float], other: float | SpatialDimension[float]) -> SpatialDimension[float]: ...

    def __sub__(self: SpatialDimension[T], other: T | float | SpatialDimension) -> SpatialDimension:
        """Subtract SpatialDimension or numeric other to SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(self.z - other.z, self.y - other.y, self.x - other.x)
        return SpatialDimension(self.z - other, self.y - other, self.x - other)

    def __rsub__(self: SpatialDimension[T], other: T | float | SpatialDimension) -> SpatialDimension:  # type: ignore[misc]
        """Right subtract SpatialDimension or numeric other to SpatialDimension."""
        if isinstance(other, SpatialDimension):
            return SpatialDimension(other.z - self.z, other.y - self.y, other.x - self.x)
        return SpatialDimension(other - self.z, other - self.y, other - self.x)
