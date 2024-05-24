"""L2 Squared Norm."""

import torch

from mrpro.operators._Functional import ProximableFunctional


class L2NormSquared(ProximableFunctional):
    """Functional for L2 Norm Squared.

    Parameters
    ----------
        lambda = 1
    """

    def __init__(self, lam: float = 1.0, g: torch.Tensor = torch.tensor([0]), dim: tuple = (None, None)):
        """init.

        Args:
            lam (float, optional): lambda scaling factor. Defaults to 1.0.
            g (torch.Tensor, optional): scaling factor. Defaults to torch.tensor([0]).
            dim (tuple, optional): dimension parameter. Defaults to (None).
        """
        super().__init__(lam=lam)
        self.g = g
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Forward method.

        Args:
            x (torch.Tensor): data tensor

        Returns
        -------
            tuple[torch.Tensor]: l2 squared of data
        """
        return (torch.pow(torch.linalg.norm(x.flatten(), ord=2, dim=self.dim, keepdim=True), 2),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L2 Norm Squared.

        Args:
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox of data
        """
        x_out = (x + sigma * self.g) / (1 + 2 * self.lam * sigma)
        return (x_out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Convex conjugate of L2 Norm Squared.

        Args:
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: convex conjugate of data
        """
        x_out = (x - sigma * self.g) / (1 + 0.5 / self.lam * sigma)
        return (x_out,)
