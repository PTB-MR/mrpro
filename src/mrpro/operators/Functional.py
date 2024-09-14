"""Base Class Functional."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from mrpro.operators.Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class."""

    def __init__(
        self,
        weight: torch.Tensor | complex = 1.0,
        target: torch.Tensor | None = None,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = False,
    ) -> None:
        r"""Initialize a Functional.

        We assume that functionals are given in the form
            :math:`f(x) = \phi( weight ( x - target))`
        for some functional :math:`phi`.

        Parameters
        ----------
        weight
            weighting of the norm (see above)
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
            target = torch.tensor(0, dtype=torch.float32)
        self.register_buffer('target', target)
        if isinstance(dim, int):
            dim = (dim,)
        elif isinstance(dim, Sequence):
            dim = tuple(dim)
        self.dim = dim
        self.divide_by_n = divide_by_n
        self.keepdim = keepdim

    def _divide_by_n(self, x: torch.Tensor, shape: None | Sequence[int]) -> torch.Tensor:
        """Compute factor for normalization."""
        if not self.divide_by_n:
            return x
        if shape is None:
            shape = x.shape
        if self.dim is not None:
            size = [shape[i] for i in self.dim]
        else:
            size = list(shape)
        return x / math.prod(size)


class ProximableFunctional(Functional, ABC):
    """ProximableFunction Base Class."""

    @abstractmethod
    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float) -> tuple[torch.Tensor]:
        """Apply proximal operator."""

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float) -> tuple[torch.Tensor]:
        """Apply proximal of convex conjugate of functional."""
        sigma = torch.as_tensor(sigma, device=self.target.device)
        sigma[sigma.abs() < 1e-8] += 1e-6
        return (x - sigma * self.prox(x * 1 / sigma, 1 / sigma)[0],)
