"""Base class for data objects."""

import dataclasses
from abc import ABC

import torch
from typing_extensions import Any

from mrpro.data.MoveDataMixin import MoveDataMixin


@dataclasses.dataclass(slots=True, frozen=True)
class Data(MoveDataMixin, ABC):
    """A general data class with field data and header."""

    data: torch.Tensor
    """Data. Shape (...other coils k2 k1 k0)"""

    header: Any
    """Header information for data."""
