from abc import ABC
from abc import abstractmethod

import torch

from mrpro.data import IData
from mrpro.data import KData


class Reconstruction(torch.nn.Module, ABC):
    """A Reconstruction."""

    @abstractmethod
    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction."""

    # Required for type hinting
    def __call__(self, kdata: KData) -> IData:
        """Apply the reconstruction."""
        return super().__call__(kdata)
