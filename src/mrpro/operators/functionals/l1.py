"""L1 Norm."""

import torch

from mrpro.operators.Functional import ProximableFunctional


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
        return (torch.tensor([(self.weight*(x - self.target)).abs().sum(dim=self.dim)]),)

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
        # diff = self.weight*(x - self.target)
        # diff = torch.view_as_real(diff)
        # return (torch.tensor([(diff).abs().sum(self.dim)]),)
        return ((torch.tensor([(self.weight*(x.real - self.target.real)).abs().sum(self.dim)]) + torch.tensor([(self.weight*(x.imag - self.target.imag)).abs().sum(self.dim)])),)

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
        # diff = x - self.target
        # threshold = torch.tensor([self.weight * sigma])
        # is_complex = diff.is_complex()
        # if is_complex:
        #     diff = torch.view_as_real(diff)
        #     threshold = torch.tensor([self.weight * sigma]).unsqueeze(-1)
        # else:
        #     threshold = torch.tensor([self.weight * sigma])
        # out = torch.nn.ReLU()(diff - threshold) - torch.nn.ReLU()(-diff - threshold)
        # if is_complex:
        #     out = torch.view_as_complex(out)
        # return (out,)
        diff_real = x.real - self.target.real
        diff_imag = x.imag - self.target.imag
        threshold = torch.tensor([self.weight * sigma])
        prox_real = torch.sign(diff_real) * torch.nn.ReLU()(diff_real.abs() - threshold)
        #prox_real = torch.nn.ReLU()(diff_real - threshold) - torch.nn.ReLU()(- diff_real - threshold)
        prox_imag = torch.sign(diff_imag) * torch.nn.ReLU()(diff_imag.abs() - threshold)
        #prox_imag = torch.nn.ReLU()(diff_imag - threshold) - torch.nn.ReLU()(- diff_imag - threshold)
        return (torch.complex(prox_real, prox_imag),)

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
        # diff = x - sigma * self.target
        # is_complex = diff.is_complex()
        # if is_complex:
        #     diff = torch.view_as_real(diff)
        #     complex_weight = torch.tensor([self.weight]).unsqueeze(-1)
        #     denom = torch.clamp(diff, -complex_weight, complex_weight)
        #     num = complex_weight * diff
        #     out = torch.view_as_complex(num / denom)
        # else:
        #     denom = torch.clamp(diff, -self.weight, self.weight)
        #     num = self.weight * diff
        #     out = num / denom
        # return (out,)
        diff_real = x.real - sigma * self.target.real
        diff_imag = x.imag - sigma * self.target.imag
        clamp_real = torch.clamp(diff_real, -self.weight, self.weight)
        clamp_imag = torch.clamp(diff_imag, -self.weight, self.weight)
        #num_real = (self.weight * diff_real)
        #num_imag = (self.weight * diff_imag)
        return (torch.complex(clamp_real, clamp_imag),)