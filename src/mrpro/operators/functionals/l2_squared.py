"""L2 Squared Norm."""

import torch

from mrpro.operators.Functional import ProximableFunctional


class L2NormSquared(ProximableFunctional):
    """Functional for L2 Norm squared.

    Parameters
    ----------
        weight = 1
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Forward method.

        Parameters
        ----------
            x
                data tensor

        Returns
        -------
            l2 norm squared of data
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        if self.divide_by_n:
            return ((torch.linalg.vector_norm(self.weight * (x - target), ord=2, dim=self.dim, keepdim=False).square())/x.numel(),)
        else:
            return (torch.linalg.vector_norm(self.weight * (x - target), ord=2, dim=self.dim, keepdim=False).square(),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L2 Norm Squared.

        Parameters
        ----------
            x
                data tensor
            sigma
                scaling factor

        Returns
        -------
            Proximal of data
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        x_out = (x + sigma * target) / (1 + 2 * self.weight * sigma)
        return (x_out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Convex conjugate of L2 Norm Squared.

        Parameters
        ----------
            x
                data tensor
            sigma
                scaling factor

        Returns
        -------
            Proximal of convex conjugate of data
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        x_out = (x - sigma * target) / (1 + 0.5 / self.weight * sigma)
        return (x_out,)
