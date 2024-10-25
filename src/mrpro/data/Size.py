"""A class to represent the size of data"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Self

import torch


@dataclass(slots=True, frozen=True)
class Shape:
    """A class to represent the size of data."""

    other: torch.Size | int | None
    """Size or Sizes along the `other` dimension"""

    coil: int | None
    """Size along the `coil` dimension"""

    dim2: int | None
    """Size along the 2. (`k2` or `z`) dimension"""

    dim1: int | None
    """Size along the 1. (`k1` or `y`) dimension"""

    dim0: int | None
    """Size along the 0. (`k0` or `x`) dimension"""

    __tuple: tuple[int, ...] | None = None
    """Used as cache"""

    @property
    def astuple(self) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        if self.__tuple is None:
            flat = []
            for name in self.__slots__():
                value = getattr(self, name)
                if value is None:
                    continue
                if isinstance(value, int):
                    flat.append(value)
                elif isinstance(value, Sequence):
                    flat.extend(value)
            self.__tuple = tuple(flat)
        return self.__tuple

    def __len__(self) -> int:
        """Return the number of dimensions in the shape."""
        return len(self.astuple)

    def __getitem__(self, idx: int | str | slice) -> torch.Size:
        """Return a specific dimension of the shape."""
        if isinstance(self, int) and -len(self) <= idx < len(self) and isinstance(idx, slice):
            result = self.astuple[idx]
        if isinstance(idx, str) and hasattr(self, idx):
            result = getattr(self, idx)
        if result is None:
            return torch.Size()
        return torch.Size(result)

    def __iter__(self) -> Iterator[int]:
        """Return Iterator."""
        return iter(self.astuple)

    def __repr__(self) -> str:
        """Return a string representation of the shape."""
        field_strings = []
        for name in self.__slots__():
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, int):
                field_strings.append(f'{name}={value}')
            elif isinstance(value, Sequence):
                field_strings.append(f'{name}={tuple(value)}')
        return 'Shape(' + ', '.join(field_strings) + ')'

    def numel(self) -> int:
        """Return the number of elements a Tensor with the given size would contain.

        See also torch.Size.numel()
        """
        return math.prod(self._flat)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Shape):
            return all(getattr(self, name) == getattr(other, name) for name in self.__slots__())
        if isinstance(other, Sequence):
            return self._flat == other
        return NotImplemented

    @classmethod
    def broadcast(cls, *shapes: 'Shape') -> Self:
        """Return the shape that results from broadcasting the given shapes."""
        instance = cls()
        if not shapes:
            return instance
        for name in cls.__slots__:
            values = [shape.other for shape in shapes if shape.other is not None]
            value = torch.broadcast_shapes(*values) if values else None
            if len(value) == 1:
                setattr(instance, name, value[0])
            else:
                setattr(instance, name, value)
