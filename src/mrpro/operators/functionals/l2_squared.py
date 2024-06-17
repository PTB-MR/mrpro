"""L2 Squared Norm."""

import torch

from mrpro.operators.Functional import ProximableFunctional


class L2NormSquared(ProximableFunctional):
    """Functional for L2 Norm Squared.

    Parameters
    ----------
        weight = 1
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Forward method.

        Parameters:
        ----------
            x (torch.Tensor): data tensor

        Returns
        -------
            tuple[torch.Tensor]: l2 squared of data
        """
        return (torch.linalg.vector_norm(self.weight * (x - self.target), ord=2, dim=self.dim, keepdim=True).square(),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L2 Norm Squared.

        Parameters:
        ----------
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox of data
        """
        x_out = (x + sigma * self.target) / (1 + 2 * self.weight * sigma)
        return (x_out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Convex conjugate of L2 Norm Squared.

        Parameters:
        ----------
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: convex conjugate of data
        """
        x_out = (x - sigma * self.target) / (1 + 0.5 / self.weight * sigma)
        return (x_out,)
