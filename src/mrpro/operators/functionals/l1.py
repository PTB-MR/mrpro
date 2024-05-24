"""L1 Norm."""

import torch

from mrpro.operators._Functional import ProximableFunctional


class L1Norm(ProximableFunctional):
    """Functional for L1 Norm.

    Parameters
    ----------
        lambda = 1
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): data tensor

        Returns
        -------
            tuple[torch.Tensor]: forward of data
        """
        return (torch.tensor([(x - self.g).abs().sum(self.dim)]),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L1 Norm.

        Args:
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox of data
        """
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            threshold = torch.tensor([self.lam * sigma]).unsqueeze(-1)
        else:
            threshold = torch.tensor([self.lam * sigma])
        x = torch.clamp(x, -threshold, threshold)
        if is_complex:
            x = torch.view_as_complex(x)
        return (x,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox convex conjugate of data.

        Args:
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox convex conjugate of data
        """
        is_complex = x.is_complex()
        if is_complex:
            x = torch.view_as_real(x)
            complex_lam = torch.tensor([self.lam]).unsqueeze(-1)
            complex_sigma = torch.tensor([sigma]).unsqueeze(-1)
            complex_g = torch.tensor([self.g]).unsqueeze(-1)
            x = torch.clamp((x - complex_g * complex_sigma), -complex_lam, complex_lam)
            x = torch.view_as_complex(x)
            return (x,)
        else:
            x = torch.clamp((x - self.g * sigma), -self.lam, self.lam)
            return (x,)
