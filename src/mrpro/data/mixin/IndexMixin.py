"""Indexing Mixin."""

from copy import copy as shallowcopy
from dataclasses import fields
from typing import Any, TypeVar, cast

import torch
from typing_extensions import Self

from mrpro.utils.indexing import Indexer
from mrpro.utils.typing import TorchIndexerType

T = TypeVar('T')


class IndexMixin:
    """Adds indexing (getitem) to a dataclass."""

    def _view(self, index: Indexer, memo=None) -> Self:
        new = shallowcopy(self)

        if memo is None:
            memo = {}

        def index_tensor(data: torch.Tensor) -> torch.Tensor:
            return index(data)

        def index_module(data: torch.nn.Module) -> torch.nn.Module:
            return data._apply(index_tensor, recurse=True)

        def index_mixin(obj: IndexMixin) -> IndexMixin:
            return obj._view(index, memo)

        def _index(data: T) -> T:
            indexed: Any  # https://github.com/python/mypy/issues/10817
            if isinstance(data, torch.Tensor):
                indexed = index_tensor(data)
            elif isinstance(data, IndexMixin):
                indexed = index_mixin(data)
            elif isinstance(data, torch.nn.Module):
                indexed = index_module(data)
            else:
                indexed = data
            return cast(T, indexed)

        new.apply_(_index, memo=memo, recurse=False)
        return new

    def __getitem__(self, *index: TorchIndexerType | Indexer) -> Self:
        index_ = index[0] if isinstance(index[0], Indexer) else Indexer(self.shape, index)
        return self._view(index_)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the dataclass."""
        shapes = []
        for elem in fields(self):
            value = getattr(self, elem.name)
            if hasattr(value, 'shape'):
                shapes.append(value.shape)
        shape = torch.broadcast_shapes(*shapes)
        return shape

    def split(self, dim:int, size:int=1, step:int=1):
        shape = self.shape
        slices = [slice(start, start + size) for start in range(0, shape[dim], step)]
        result = [self[slice] for slice in slices]
        return result

    def sliding_window(self, dim, size, stride, dilation)

