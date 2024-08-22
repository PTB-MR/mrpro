"""L1 Norm."""

import torch

from mrpro.operators.Functional import ProximableFunctional


class L1Norm(ProximableFunctional):
    """Functional for L1 Norm.

    Parameters
    ----------
        weight = 1
    """

    def forward(self, x: torch.Tensor, dim: torch.Tensor | None = None, keep_dim: bool | None = None) -> tuple[torch.Tensor]:
        """Forward method.

        Parameters
        ----------
            x
                data tensor

        Returns
        -------
            L1 norm of data
        """
        if dim is None:
            dim = self.dim
        if keep_dim is None:
            keep_dim = self.keep_dim
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        if self.divide_by_n:
            return ((self.weight * (x - target)).abs().mean(dim=dim, keepdim=keep_dim),)
        else:
            return ((self.weight * (x - target)).abs().sum(dim=dim, keepdim=keep_dim),)

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
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target
        is_complex = diff.is_complex()
        threshold = torch.tensor([self.weight * sigma])
        if self.divide_by_n and self.dim is not None:
            dim_to_keep = [d if d >= 0 else d + x.ndimension() for d in self.dim]
            n = torch.prod(torch.tensor([x.shape[i] for i in dim_to_keep]))
            threshold /= n   
        if is_complex:
            x_out = target + torch.polar(torch.nn.functional.relu(diff.abs() - threshold), torch.angle(diff))
        else:
            x_out = target + (
                torch.nn.functional.relu(diff - threshold) - torch.nn.functional.relu(-diff - threshold)
            )
        return (x_out,)

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
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - sigma * target
        is_complex = diff.is_complex()
        if is_complex:
            x_out = torch.polar(self.weight.clamp(max=diff.abs()), torch.angle(diff))
        else:
            num = self.weight * diff
            denom = torch.max(diff.abs(), self.weight)
            x_out = num / denom
        return (x_out,)


class L1NormViewAsReal(ProximableFunctional):
    """Functional for L1 Norm with real and imaginary part treated seperately.

    Parameters
    ----------
        weight = 1
    """

    def forward(self, x: torch.Tensor, dim: torch.Tensor | None = None, keep_dim: bool | None = None) -> tuple[torch.Tensor]:
        """Forward method.

        Parameters
        ----------
            x
                data tensor

        Returns
        -------
            L1 norm of data
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target
        is_complex = diff.is_complex()
        if is_complex:
            return ((L1Norm().forward(diff.real, dim=dim, keep_dim=keep_dim)[0]) + (L1Norm().forward(diff.imag, dim=dim, keep_dim=keep_dim)[0]),)
        else:
            return ((L1Norm().forward(diff, dim=dim, keep_dim=keep_dim)[0]),)

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
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target
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
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - sigma * target
        is_complex = diff.is_complex()
        if is_complex:
            return (
                torch.complex(
                    L1Norm().prox_convex_conj(diff.real, sigma)[0], L1Norm().prox_convex_conj(diff.imag, sigma)[0]
                ),
            )
        else:
            return (torch.tensor([L1Norm().prox_convex_conj(diff, sigma)[0]]),)
        