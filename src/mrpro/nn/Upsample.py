from typing import Literal

import torch
from torch.nn import Module

from mrpro.utils.interpolate import interpolate


class Upsample(Module):
    def __init__(self, dim: int, scale_factor: int = 2, mode: Literal['nearest', 'linear'] = 'linear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_size = [d * self.scale_factor for d in x.shape[self.dim :]]
        return interpolate(x, size=new_size, dim=range(-self.dim, 0))
