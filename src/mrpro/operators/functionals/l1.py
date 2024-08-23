"""L1 Norm."""

import torch
from mrpro.operators.Functional import ProximableFunctional
from collections.abc import Sequence


class L1Norm(ProximableFunctional):
    """Functional for L1 Norm.

    Parameters
    ----------
        weight = 1
    """

    def forward(self, x: torch.Tensor, dim: Sequence[int] | None = None, keepdim: bool | None = None, divide_by_n : bool | None = None) -> tuple[torch.Tensor]:
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
        if keepdim is None:
            keepdim = self.keepdim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n
            
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        
        if divide_by_n:
            return ((self.weight * (x - target)).abs().mean(dim=dim, keepdim=keepdim),)
        else:
            return ((self.weight * (x - target)).abs().sum(dim=dim, keepdim=keepdim),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor, dim: Sequence[int] | None = None, divide_by_n : bool | None = None) -> tuple[torch.Tensor]:
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
        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n
            
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target
        is_complex = diff.is_complex()
        threshold = torch.tensor([self.weight * sigma])
        
        if divide_by_n:
            n = torch.prod(torch.tensor([x.shape[i] for i in dim])) if dim is not None else x.numel()
            threshold = threshold.to(torch.float32)
            threshold /= n
            
        if is_complex:
            x_out = target + torch.polar(torch.nn.functional.relu(diff.abs() - threshold), torch.angle(diff))
        else:
            x_out = target + (
                torch.nn.functional.relu(diff - threshold) - torch.nn.functional.relu(-diff - threshold)
            )
        return (x_out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor, dim: Sequence[int] | None = None, divide_by_n : bool | None = None) -> tuple[torch.Tensor]:
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
        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n
            
        if divide_by_n:
            n = torch.prod(torch.tensor([x.shape[i] for i in dim])) if dim is not None else x.numel()
            sigma /= n
            
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
    """Functional for L1 Norm with real and imag part treated seperately.

    Parameters
    ----------
        weight = 1
    """

    def forward(self, x: torch.Tensor, dim: torch.Tensor | None = None, keepdim: bool | None = None, divide_by_n : bool | None = None) -> tuple[torch.Tensor]:
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
        if keepdim is None:
            keepdim = self.keepdim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n
            
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target
        is_complex = diff.is_complex()
        if is_complex:
            return ((L1Norm().forward(diff.real, dim=dim, keepdim=keepdim, divide_by_n=divide_by_n)[0]) + (L1Norm().forward(diff.imag, dim=dim, keepdim=keepdim, divide_by_n=divide_by_n)[0]),)
        else:
            return ((L1Norm().forward(diff, dim=dim, keepdim=keepdim, divide_by_n=divide_by_n)[0]),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor, dim: torch.Tensor | None = None, divide_by_n : bool | None = None) -> tuple[torch.Tensor]:
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
        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n
            
        if divide_by_n:
            n = torch.prod(torch.tensor([x.shape[i] for i in dim])) if dim is not None else x.numel()
            sigma /= n
            divide_by_n = False
            
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target
        is_complex = diff.is_complex()
        if is_complex:
            return (torch.complex(L1Norm().prox(diff.real, sigma, dim, divide_by_n)[0], L1Norm().prox(diff.imag, sigma, dim, divide_by_n)[0]),)
        else:
            return (torch.tensor([(L1Norm().prox(diff, sigma, dim, divide_by_n)[0])]),)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor, dim: torch.Tensor | None = None, divide_by_n : bool | None = None) -> tuple[torch.Tensor]:
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
        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n
            
        if divide_by_n:
            n = torch.prod(torch.tensor([x.shape[i] for i in dim])) if dim is not None else x.numel()
            sigma /= n
            divide_by_n = False
            
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - sigma * target
        is_complex = diff.is_complex()
        if is_complex:
            return (
                torch.complex(
                    L1Norm().prox_convex_conj(diff.real, sigma, dim, divide_by_n)[0], L1Norm().prox_convex_conj(diff.imag, sigma, dim, divide_by_n)[0]
                ),
            )
        else:
            return (torch.tensor([L1Norm().prox_convex_conj(diff, sigma, dim, divide_by_n)[0]]),)
        