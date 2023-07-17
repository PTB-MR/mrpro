from abc import ABC
from abc import abstractmethod

import torch

from mrpro.data.raw import KHeader


class KTrajectory(ABC):
    def __init__(self) -> None:
        self.traj: torch.Tensor | None = None

    @abstractmethod
    def calc_traj(self, header: KHeader) -> None:
        pass
