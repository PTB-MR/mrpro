"""Some type hints that are used in multiple places in the codebase but not part of mrpro's public API."""

from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import Any

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


else:
    from numpy._typing import _NestedSequence

    TorchIndexerType: TypeAlias = Any
    """Torch indexer type."""

    NestedSequence: TypeAlias = _NestedSequence
    """A nested sequence type."""

    NumpyIndexerType: TypeAlias = Any
    """Numpy indexer type."""

__all__ = ['NestedSequence', 'NumpyIndexerType', 'TorchIndexerType']
