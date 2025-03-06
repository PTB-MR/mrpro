"""Base class for data objects."""

from abc import ABC

import torch
from typing_extensions import Any

from mrpro.data.Dataclass import Dataclass


class Data(Dataclass, ABC):
    """A general data class with field data and header."""

    data: torch.Tensor
    """Data. Shape `(...other coils k2 k1 k0)`"""

    header: Any
    """Header information for data."""
