"""Mixin to reduce dimensions with repeated values to singleton in fields of dataclasses."""

import dataclasses
from collections.abc import Sequence
from typing import TypeVar, cast

import torch

from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.reduce_repeat import reduce_repeat as remove_repeat_tensor
from mrpro.utils.typing import DataclassInstance

T = TypeVar('T')


def remove_repeat(data: T, tol: float, dim: Sequence[int] | None = None) -> T:
    """Replace dimensions with all equal values with singletons in fields.

    Handles Tensor, Rotation, and SpatialDimension fields.

    Parameters
    ----------
    data:
        Input data, must be real.
    tol:
        tolerance.
    dim
        dimensions to try to reduce to singletons. `None` means all.
    """
    match data:
        case torch.Tensor():
            return cast(T, remove_repeat_tensor(data, tol, dim))
        case SpatialDimension(z, y, x) if (
            isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(z, torch.Tensor)
        ):
            return cast(
                T,
                data.__class__(
                    x=remove_repeat_tensor(x, tol, dim),
                    y=remove_repeat_tensor(y, tol, dim),
                    z=remove_repeat_tensor(z, tol, dim),
                ),
            )
        case Rotation():
            tensor = data.as_matrix().flatten(start_dim=-2)
            tensor = torch.stack([remove_repeat_tensor(x, tol, dim) for x in tensor.unbind(-1)], -1).unflatten(
                -1, (3, 3)
            )
            return cast(T, data.__class__.from_matrix(tensor))
        case _:
            return data


@dataclasses.dataclass
class ReduceRepeatMixin(DataclassInstance):
    """Adds a __post_init__ method to remove repeated dimensions fields."""

    def __init_subclass__(cls, reduce_repeat: bool = True, **kwargs):
        """Initialize a checked data subclass."""
        super().__init_subclass__(**kwargs)
        if reduce_repeat:
            # inject the new post_init method
            original_post_init = vars(cls).get('__post_init__')

            def new_post_init(self: ReduceRepeatMixin) -> None:
                for field in dataclasses.fields(self):
                    setattr(self, field.name, remove_repeat(getattr(self, field.name), 1e-6))
                if original_post_init is not None:
                    original_post_init(self)

            cls.__post_init__ = new_post_init  # type: ignore[attr-defined]
