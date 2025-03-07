"""Mixin for Indexing of Data"""

from typing_extensions import Self

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.utils.indexing import Indexer
from mrpro.utils.typing import TorchIndexerType


class IndexMixin(MoveDataMixin):
    """Mixin for Indexing of Data"""

    def __getitem__(self, index: TorchIndexerType) -> Self:
        """Indexing of Data"""
        indexer = Indexer(self.shape, index)
        self.apply_(indexer)
