"""Some type hints that are used in multiple places in the codebase but not part of mrpro's public API."""

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from types import EllipsisType
    from typing import SupportsIndex, TypeAlias

    import torch
    from torch._C import _NestedSequence as NestedSequence

    # This matches the torch.Tensor indexer typehint
    _IndexerTypeInner: TypeAlias = None | bool | int | slice | EllipsisType | torch.Tensor
    _SingleIndexerType: TypeAlias = SupportsIndex | _IndexerTypeInner | NestedSequence[_IndexerTypeInner]
    IndexerType: TypeAlias = tuple[_SingleIndexerType, ...] | _SingleIndexerType
else:
    IndexerType: TypeAlias = Any
    NestedSequence: TypeAlias = Any
