"""SpatialDimension dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
import torch
from numpy.typing import ArrayLike

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

    z: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor

    def __init__(self, z: T, y: T, x: T):
        """Create a SpatialDimension from torch.tensors, float or int values.

        Parameters
        ----------
        z
            spatial dimension along z direction
        y
            spatial dimension along y direction
        x
            spatial dimension along x direction
        """
        self.z = torch.as_tensor(z)
        self.y = torch.as_tensor(y)
        self.x = torch.as_tensor(x)

    @classmethod
    def from_xyz(cls, data: XYZ[T], conversion: Callable[[T], T] | None = None) -> SpatialDimension[T]:
        """Create a SpatialDimension from something with (.x .y .z) parameters.

        Parameters
        ----------
        data
            should implement .x .y .z. For example ismrmrd's matrixSizeType.
        conversion,  optional
            will be called for each value to convert it
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
            will be called for each value to convert it
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
            will be called for each value to convert it
        """
        data = torch.flip(torch.as_tensor(data), (-1,))
        return SpatialDimension.from_array_xyz(data, conversion)

    @property
    def zyx(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a z,y,x tuple."""
        return (self.z, self.y, self.x)

    @property
    def xyz(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a x,y,z tuple."""
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        """Return a string representation of the SpatialDimension."""
        return f'z={self.z}, y={self.y}, x={self.x}'

    def __getitem__(self, idx: Any) -> SpatialDimension:
        """Get SpatialDimension item."""
        return SpatialDimension(self.z[idx], self.y[idx], self.x[idx])

    def __setitem__(self, idx: Any, value: SpatialDimension):
        """Set SpatialDimension item."""
        self.z[idx] = value.z
        self.y[idx] = value.y
        self.x[idx] = value.x

    def __mul__(self, value: float | int | SpatialDimension) -> SpatialDimension:
        """Multiply SpatialDimension with numeric value or SpatialDimension."""
        if isinstance(value, SpatialDimension):
            return SpatialDimension(self.z * value.z, self.y * value.y, self.x * value.x)
        return SpatialDimension(self.z * value, self.y * value, self.x * value)

    def __rmul__(self, value: float | int | SpatialDimension) -> SpatialDimension:
        """Right multiply SpatialDimension with numeric value or SpatialDimension."""
        return self.__mul__(value)

    def __truediv__(self, value: float | int | SpatialDimension) -> SpatialDimension:
        """Divide SpatialDimension with numeric value or SpatialDimension."""
        if isinstance(value, SpatialDimension):
            return SpatialDimension(self.z / value.z, self.y / value.y, self.x / value.x)
        return SpatialDimension(self.z / value, self.y / value, self.x / value)

    def __rtruediv__(self, value: float | int) -> SpatialDimension:
        """Right divide SpatialDimension with numeric value."""
        return SpatialDimension(value / self.z, value / self.y, value / self.x)

    def __add__(self, value: float | int | SpatialDimension) -> SpatialDimension:
        """Add SpatialDimension or numeric value to SpatialDimension."""
        if isinstance(value, SpatialDimension):
            return SpatialDimension(self.z + value.z, self.y + value.y, self.x + value.x)
        return SpatialDimension(self.z + value, self.y + value, self.x + value)

    def __radd__(self, value: float | int | SpatialDimension) -> SpatialDimension:
        """Right add SpatialDimension or numeric value to SpatialDimension."""
        return self.__add__(value)

    def __sub__(self, value: float | int | SpatialDimension) -> SpatialDimension:
        """Subtract SpatialDimension or numeric value to SpatialDimension."""
        if isinstance(value, SpatialDimension):
            return SpatialDimension(self.z - value.z, self.y - value.y, self.x - value.x)
        return SpatialDimension(self.z - value, self.y - value, self.x - value)

    def __rsub__(self, value: float | int | SpatialDimension) -> SpatialDimension:
        """Right subtract SpatialDimension or numeric value to SpatialDimension."""
        if isinstance(value, SpatialDimension):
            return SpatialDimension(value.z - self.z, value.y - self.y, value.x - self.x)
        return SpatialDimension(value - self.z, value - self.y, value - self.x)
