"""Base Class Functional."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from mrpro.operators.Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class."""

    def __init__(
        self,
        weight: torch.Tensor | float = 1.0,
        target: torch.Tensor | None = None,
        dim: Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = True,
    ) -> None:
        r"""Initialize a Functional.

        We assume that functionals are given in the form
            f(x) = \phi( weight ( x - target))
        for some functional phi.

        Parameters
        ----------
            weight
                weighting of the norm
            target
                element to which distance is taken - often data tensor
            dim
                dimension over which norm is calculated
            divide_by_n
                True: norm calculated with mean
                False: norm calculated with sum
            keepdim
                whether or not to maintain the dimensions of the input

        """
        super().__init__()
        self.register_buffer('weight', torch.as_tensor(weight))
        if target is None:
            target = torch.tensor([0.0], dtype=torch.float32)
        self.register_buffer('target', target)
        self.dim = dim
        self.divide_by_n = divide_by_n
        self.keepdim = keepdim


class ProximableFunctional(Functional, ABC):
    """ProximableFunction Base Class."""

    @abstractmethod
    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply proximal operator."""

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply proximal of convex conjugate of functional."""
        return (x - sigma * self.prox(x * 1 / sigma, 1 / sigma),)
