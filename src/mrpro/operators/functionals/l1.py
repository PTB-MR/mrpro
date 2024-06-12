"""L1 Norm."""

import torch

from mrpro.operators._Functional import ProximableFunctional


class L1Norm(ProximableFunctional):
    """Functional for L1 Norm.

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
            tuple[torch.Tensor]: forward of data
        """
        return (torch.tensor([(self.weight*(x - self.target)).abs().sum(self.dim)]),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L1 Norm.

        Parameters:
        ----------
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox of data
        """
        diff = x - self.target
        threshold = torch.tensor([self.weight * sigma])
        out = torch.polar(torch.nn.ReLU()(diff.abs() - threshold), torch.angle(diff))
        return (out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox convex conjugate of data.

        Parameters:
        ----------
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox convex conjugate of data
        """
        diff = x - sigma * self.target
        out = torch.polar(self.weight.clamp(max = diff.abs()), torch.angle(diff))
        return (out,)


class L1NormComponentwise(ProximableFunctional):
    """Functional for L1 Norm componentwise.

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
            tuple[torch.Tensor]: forward of data
        """
        return ((torch.tensor([(self.weight*(x - self.target)).real().sum(self.dim)]) + torch.tensor([(self.weight*(x - self.target)).imag().sum(self.dim)])),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L1 Norm.

        Parameters:
        ----------
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox of data
        """
        diff = x - self.target
        threshold = torch.tensor([self.weight * sigma])
        is_complex = diff.is_complex()
        if is_complex:
            diff = torch.view_as_real(diff)
            threshold = torch.tensor([self.weight * sigma]).unsqueeze(-1)
        else:
            threshold = torch.tensor([self.weight * sigma])
        out = torch.nn.ReLU()(diff - threshold) - torch.nn.ReLU()(-diff - threshold)
        if is_complex:
            out = torch.view_as_complex(out)
        return (out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox convex conjugate of data.

        Parameters:
        ----------
            x (torch.Tensor): data tensor
            sigma (torch.Tensor): scaling factor

        Returns
        -------
            tuple[torch.Tensor]: prox convex conjugate of data
        """
        diff = x - sigma * self.target
        is_complex = diff.is_complex()
        if is_complex:
            diff = torch.view_as_real(diff)
            complex_weight = torch.tensor([self.weight]).unsqueeze(-1)
            denom = torch.clamp(diff, -complex_weight, complex_weight)
            num = complex_weight * diff
            out = torch.view_as_complex(num / denom)
        else:
            denom = torch.clamp(diff, -self.weight, self.weight)
            num = self.weight * diff
            out = num / denom
        return (out,)