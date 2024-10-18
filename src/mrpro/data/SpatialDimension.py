"""SpatialDimension dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

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
    def zyx(self) -> tuple[T, T, T]:
        """Return a z,y,x tuple."""
        return (self.z, self.y, self.x)

    def __str__(self) -> str:
        """Return a string representation of the SpatialDimension."""
        return f'z={self.z}, y={self.y}, x={self.x}'
