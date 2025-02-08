"""Some type hints that are used in multiple places in the codebase but not part of mrpro's public API."""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import Any, ParamSpec, Protocol, TypeVar, Unpack, overload

if TYPE_CHECKING:
    from types import EllipsisType
    from typing import TypeAlias

    import torch
    from numpy import ndarray
    from numpy._typing import _NestedSequence as NestedSequence
    from typing_extensions import SupportsIndex

    # This matches the torch.Tensor indexer typehint
    _TorchIndexerTypeInner: TypeAlias = None | bool | int | slice | EllipsisType | torch.Tensor
    _SingleTorchIndexerType: TypeAlias = SupportsIndex | _TorchIndexerTypeInner | NestedSequence[_TorchIndexerTypeInner]
    TorchIndexerType: TypeAlias = tuple[_SingleTorchIndexerType, ...] | _SingleTorchIndexerType

    # This matches the numpy.ndarray indexer typehint
    _SingleNumpyIndexerType: TypeAlias = ndarray | SupportsIndex | None | slice | EllipsisType
    NumpyIndexerType: TypeAlias = tuple[_SingleNumpyIndexerType, ...] | _SingleNumpyIndexerType

    Tout = TypeVar('Tout', bound=tuple[torch.Tensor, ...], covariant=True)

    P = ParamSpec('P')
    Wrapped: TypeAlias = Callable[P, Tout]

    F = TypeVar('F', bound=Wrapped)

    class _EndomorphCallable(Protocol):
        """A callable with the same number of tensor inputs and outputs.

        This is a protocol for a callable that takes a variadic number of tensor inputs
        and returns the same number of tensor outputs.

        This is only implemented for up to 10 inputs, if more inputs are given, the return
        will be a variadic number of tensors.

        This Protocol is used to decorate functions with the `endomorph` decorator.
        """

        @overload
        def __call__(self, /, **kwargs) -> tuple[()]: ...
        @overload
        def __call__(self, x1: torch.Tensor, /, **kwargs) -> tuple[torch.Tensor]: ...

        @overload
        def __call__(self, x1: torch.Tensor, x2: torch.Tensor, /, **kwargs) -> tuple[torch.Tensor, torch.Tensor]: ...

        @overload
        def __call__(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, /, **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

        @overload
        def __call__(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, /, **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

        @overload
        def __call__(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor, /, **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

        @overload
        def __call__(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            /,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

        @overload
        def __call__(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            /,
            **kwargs,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]: ...

        @overload
        def __call__(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
            /,
            **kwargs,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]: ...

        @overload
        def __call__(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
            x9: torch.Tensor,
            /,
            **kwargs,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]: ...

        @overload
        def __call__(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
            x9: torch.Tensor,
            x10: torch.Tensor,
            /,
            **kwargs,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]: ...

        @overload
        def __call__(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
            x4: torch.Tensor,
            x5: torch.Tensor,
            x6: torch.Tensor,
            x7: torch.Tensor,
            x8: torch.Tensor,
            x9: torch.Tensor,
            x10: torch.Tensor,
            /,
            *args: torch.Tensor,
            **kwargs,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Unpack[tuple[torch.Tensor, ...]],
        ]: ...

        @overload
        def __call__(self, /, *args: torch.Tensor, **kwargs) -> tuple[torch.Tensor, ...]: ...

        def __call__(self, /, *args: torch.Tensor, **kwargs) -> tuple[torch.Tensor, ...]:
            """Apply the Operator."""

    def endomorph(f: F, /) -> _EndomorphCallable:
        """Decorate a function to make it an endomorph callable.

        This adds overloads for N->N-Tensor signatures, for N<10.
        For >10 inputs, the return type will a tuple with >10 tensors.
        """
        return f

else:
    TorchIndexerType: TypeAlias = Any
    """Torch indexer type."""

    class NestedSequence(Protocol[TypeVar('T')]):
        """A nested sequence type."""

        ...

    NumpyIndexerType: TypeAlias = Any
    """Numpy indexer type."""

    def endomorph(f: Callable) -> Callable:
        """Decorate a function to make it an endomorph callable."""
        return f


__all__ = ['NestedSequence', 'NumpyIndexerType', 'TorchIndexerType', 'endomorph']
