"""L2 Squared Norm."""

from collections.abc import Sequence

import torch

from mrpro.operators.Functional import ProximableFunctional


class L2NormSquared(ProximableFunctional):
    r"""Functional class for the squared L2 Norm.

    This implements the functional given by
        f: C^N --> [0, \infty), x ->  1/2 * \| W*(x-b)\|_2^2,
    where W is a either a scalar or tensor that corresponds to a (block-) diagonal operator
    that is applied to the input. This is, for example, useful for non-Cartesian MRI
    reconstruction when     using a density-compensation function for k-space pre-conditioning.

    The norm of the vector is computed along the dimensions given by "dim".

    Further, it is possible to scale the functional by N, i.e. by the number voxels of
    the elements of the vector space that is spanned by the dimensions indexed by "dim".
    If "dim" is set to None and "keepdim" to False, the result is a single number, which
    is typically of interest for computing loss functions.

    Further, the proximal mapping and the proximal mapping of the convex conjugate
    """

    def forward(
        self,
        x: torch.Tensor,
        dim: Sequence[int] | None = None,
        divide_by_n: bool | None = None,
        keepdim: bool | None = None,
    ) -> tuple[torch.Tensor]:
        """Forward method.

        Compute the squared l2-norm of the input.

        Parameters
        ----------
            x
                input tensor
            dim
                dimensions that span the considered vector space
            divide_by_n
                divide by the number of voxels or not
            keepdim
                whether or not to maintain the dimensions of the input

        Returns
        -------
            squared l2 norm of the input tensor
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)

        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n
        if keepdim is None:
            keepdim = self.keepdim

        if divide_by_n:
            return (0.5 * (self.weight * (x - target)).abs().pow(2).mean(self.dim, keepdim=keepdim),)
        else:
            return (0.5 * (self.weight * (x - target)).abs().pow(2).sum(self.dim, keepdim=keepdim),)

    def prox(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        dim: Sequence[int] | None = None,
        divide_by_n: bool | None = None,
    ) -> tuple[torch.Tensor]:
        """Proximal Mapping of the Squared L2 Norm.

        Compute the proximal mapping of the squared l2-norm.

        Parameters
        ----------
            x
                input tensor
            sigma
                scaling factor
            dim
                dimensions that span the considered vector space
            divide_by_n
                divide by the number of voxels or not

        Returns
        -------
            Proximal mapping applied to the input tensor
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)

        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n

        if divide_by_n:
            n = torch.prod(torch.tensor([x.shape[i] for i in dim])) if dim is not None else x.numel()
            sigma /= n

        x_out = (x + sigma * self.weight.conj() * self.weight * target) / (1 + sigma * self.weight.conj() * self.weight)

        return (x_out,)

    def prox_convex_conj(
        self, x: torch.Tensor, sigma: torch.Tensor, dim: Sequence[int] | None = None, divide_by_n: bool | None = None
    ) -> tuple[torch.Tensor]:
        """Convex conjugate of L2 Norm Squared.

        Compute the proximal mapping of the convex conjugate of the squared l2-norm.

        Parameters
        ----------
            x
                data tensor
            sigma
                scaling factor
            dim
                dimensions that span the considered vector space
            divide_by_n
                divide by the number of voxels or not

        Returns
        -------
            Proximal of convex conjugate of data
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)

        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n

        if divide_by_n:
            n = torch.prod(torch.tensor([x.shape[i] for i in dim])) if dim is not None else x.numel()
            factor = n
        else:
            factor = 1.0

        x_out = (x - sigma * target) / (1 + sigma * factor / (self.weight * self.weight.conj()))
        return (x_out,)
