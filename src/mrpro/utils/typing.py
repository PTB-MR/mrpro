"""Some type hints that are used in multiple places in the codebase but not part of mrpro's public API."""

from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import Any

if TYPE_CHECKING:
    from types import EllipsisType
    from typing import TypeAlias

    import torch
    from numpy import ndarray
    from torch._C import _NestedSequence as NestedSequence
    from typing_extensions import SupportsIndex

    # This matches the torch.Tensor indexer typehint
    _TorchIndexerTypeInner: TypeAlias = None | bool | int | slice | EllipsisType | torch.Tensor
    _SingleTorchIndexerType: TypeAlias = SupportsIndex | _TorchIndexerTypeInner | NestedSequence[_TorchIndexerTypeInner]
    TorchIndexerType: TypeAlias = tuple[_SingleTorchIndexerType, ...] | _SingleTorchIndexerType

    # This matches the numpy.ndarray indexer typehint
    _SingleNumpyIndexerType: TypeAlias = ndarray | SupportsIndex | None | slice | EllipsisType
    NumpyIndexerType: TypeAlias = tuple[_SingleNumpyIndexerType, ...] | _SingleNumpyIndexerType


else:
    TorchIndexerType: TypeAlias = Any
    NestedSequence: TypeAlias = Any
    NumpyIndexerType: TypeAlias = Any

__all__ = ['NestedSequence', 'NumpyIndexerType', 'TorchIndexerType']
