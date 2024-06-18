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

        Parameters
        ----------
            x
                data tensor

        Returns
        -------
            L1 norm of data
        """
        return (torch.tensor([(self.weight * (x - self.target)).abs().sum(dim=self.dim)]),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L1 Norm.

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
        diff = x - self.target
        is_complex = diff.is_complex()
        threshold = torch.tensor([self.weight * sigma])
        if is_complex:
            out = self.target + torch.polar(torch.nn.functional.relu(diff.abs() - threshold), torch.angle(diff))
        else:
            out = self.target + (
                torch.nn.functional.relu(diff - threshold) - torch.nn.functional.relu(-diff - threshold)
            )
        return (out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox convex conjugate of data.

        Parameters
        ----------
            x
                data tensors
            sigma
                scaling factor

        Returns
        -------
            Proximal of convex conjugate of data
        """
        diff = x - sigma * self.target
        is_complex = diff.is_complex()
        if is_complex:
            out = torch.polar(self.weight.clamp(max=diff.abs()), torch.angle(diff))
        else:
            num = self.weight * diff
            denom = torch.max(diff.abs(), self.weight)
            out = num / denom
        return (out,)


class L1NormViewAsReal(ProximableFunctional):
    """Functional for L1 Norm with real and imaginary part treated seperately.

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
            L1 norm of data
        """
        diff = x - self.target
        is_complex = diff.is_complex()
        if is_complex:
            return (torch.tensor([(L1Norm().forward(diff.real)[0]) + (L1Norm().forward(diff.imag)[0])]),)
        else:
            return (torch.tensor([(L1Norm().forward(diff)[0])]),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox of L1 Norm.

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
        diff = x - self.target
        is_complex = diff.is_complex()
        if is_complex:
            return (torch.complex(L1Norm().prox(diff.real, sigma)[0], L1Norm().prox(diff.imag, sigma)[0]),)
        else:
            return (torch.tensor([(L1Norm().prox(diff, sigma)[0])]),)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor]:
        """Prox convex conjugate of data.

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
        diff = x - sigma * self.target
        is_complex = diff.is_complex()
        if is_complex:
            return (
                torch.complex(
                    L1Norm().prox_convex_conj(diff.real, sigma)[0], L1Norm().prox_convex_conj(diff.imag, sigma)[0]
                ),
            )
        else:
            return (torch.tensor([L1Norm().prox_convex_conj(diff, sigma)[0]]),)
