"""Base Class Functional."""

import torch

from mrpro.operators._Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class.

    Args:
        Operator (_type_): _description_
    """

    def __init__(self, lam: float = 1.0):
        """init.

        Args:
            lam (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__()
        self.lam = lam


class ProximableFunctional(Functional):
    """ProximableFunction Base Class.

    Args:
        Functional (_type_): _description_
    """

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward operator."""
        return (x - self.prox_convex_conj(x, sigma),)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward operator."""
        return (x - self.prox(x, sigma),)
