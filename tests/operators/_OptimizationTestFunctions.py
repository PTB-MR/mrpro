"""Test functions for non-linear optimization."""

import torch
from mrpro.operators import Operator


# TODO: Consider introducing the concept of a "Functional" for scalar-valued operators
class Rosenbrock(Operator[torch.Tensor, torch.Tensor, tuple[torch.Tensor,]]):
    def __init__(self, a: float = 1, b: float = 100) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor,]:
        fval = (self.a - x1) ** 2 + self.b * (x1 - x2**2) ** 2

        return (fval,)
