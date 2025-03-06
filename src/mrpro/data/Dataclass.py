import dataclasses
from collections.abc import Callable

from typing_extensions import Any, Self, dataclass_transform

from mrpro.data.mixin.MoveDataMixin import MoveDataMixin


@dataclass_transform(frozen_default=False)
class Dataclass(MoveDataMixin):
    def __init_subclass__(cls, *args, **kwargs):
        dataclasses.dataclass(cls)
        super().__init_subclass__(**kwargs)

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
            If `True`, the function will be applied to all children that are `MoveDataMixin` instances.
        """
        new = self.clone().apply_(function, recurse=recurse)
        return new

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
            If `True`, the function will be applied to all children that are `MoveDataMixin` instances.
        """
        applied: Any

        if memo is None:
            memo = {}

        if function is None:
            return self

        for name, data in self._items():
            if id(data) in memo:
                # this works even if self is frozen
                object.__setattr__(self, name, memo[id(data)])
                continue
            if recurse and isinstance(data, MoveDataMixin):
                applied = data.apply_(function, memo=memo)
            else:
                applied = function(data)
            memo[id(data)] = applied
            object.__setattr__(self, name, applied)
        return self
