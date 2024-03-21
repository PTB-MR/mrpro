from abc import ABC
from abc import abstractmethod

import torch

from mrpro.data import IData
from mrpro.data import KData
from mrpro.utils import DataBufferMixin


class Reconstruction(DataBufferMixin, torch.nn.Module, ABC):
    """A Reconstruction."""

    @abstractmethod
    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction."""
