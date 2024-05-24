"""Base Class Functional."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence

import torch

from mrpro.operators._Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class.

    Args:
        Operator (_type_): _description_
    """

    def __init__(
        self, lam: torch.Tensor | float = 1.0, g: torch.Tensor | None = None, dim: Sequence[int] | None = None
    ):
        """init.

        Args:
            lam (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__()
        self.register_buffer('lam', torch.as_tensor(lam))
        if g is None:
            g = torch.tensor(0.0)
        self.register_buffer('g', g)
        self.dim = dim


class ProximableFunctional(Functional, ABC):
    """ProximableFunction Base Class.

    Args:
        Functional (_type_): _description_
    """

    @abstractmethod
    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward operator."""

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward operator."""
        return (x - sigma * self.prox(x * 1 / sigma, 1 / sigma),)
