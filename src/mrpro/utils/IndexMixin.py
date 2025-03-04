"""Mixin for Indexing of Data"""

import torch
from mrpro.utils.indexing import Indexer
from mrpro.utils.typing import TorchIndexerType
from mrpro.data.MoveDataMixin import MoveDataMixin
from typing_extensions import Self


class IndexMixin(MoveDataMixin):
    """Mixin for Indexing of Data"""

    def __getitem__(self, index: TorchIndexerType) -> Self:
        """Indexing of Data"""
        indexer = Indexer(self.shape, index)
        self.apply_(indexer)
