"""A parsed Indexer."""

from dataclasses import dataclass

from mrpro.utils.typing import IndexerType

COLON = slice(None)


@dataclass(slots=True, frozen=True)
class Indexer:
    """A parsed Indexer."""

    other: IndexerType = Ellipsis
    coil: IndexerType = COLON
    dim2: IndexerType = COLON
    dim1: IndexerType = COLON
    dim0: IndexerType = COLON
