"""L1 Norm."""

from collections.abc import Sequence

import torch

from mrpro.operators.Functional import ProximableFunctional


class L1NormViewAsReal(ProximableFunctional):
    r"""Functional class for the L1 Norm, where C is identified with R^2.

    This implements the functional given by
        f: C^N --> [0, \infty), x ->  \| W*(x-b)\|_1 := \|Re( W*(x-b) )\|_1 + \|Im( W*(x-b) )\|_1
    where W is a either a scalar or tensor that corresponds to a (block-) diagonal operator
    that is applied to the input.

    The norm of the vector is computed along the dimensions given by "dim".

    Further, it is possible to scale the functional by N, i.e. by the number voxels of
    the elements of the vector space that is spanned by the dimensions indexed by "dim".
    If "dim" is set to None and "keepdim" to False, the result is a single number, which
    is typically of interest for computing loss functions.

    Further, the proximal mapping and the proximal mapping of the convex conjugate are given.

    Note that here, the proximal mapping and the convex conjugate of the proximal mapping are
    derived assuming the weight vector defines an invertible (block-) diagonal
    operator with all entries being strictly greater than zero, i.e. only real-valued
    weight-tensors are allowed.
    """

    def forward(
        self,
        x: torch.Tensor,
        dim: Sequence[int] | None = None,
        keepdim: bool | None = None,
        divide_by_n: bool | None = None,
    ) -> tuple[torch.Tensor]:
        """Forward method.

        Compute the l1-norm of the input with C identified as R^2.

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
            l1 norm of the input tensor
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

        l1norm = L1Norm(weight=self.weight, target=self.target, divide_by_n=divide_by_n)

        if is_complex:
            l1norm_real = L1Norm(weight=self.weight, target=self.target.real, divide_by_n=divide_by_n)
            l1norm_imag = L1Norm(weight=self.weight, target=self.target.imag, divide_by_n=divide_by_n)

            return (
                l1norm_real.forward(diff.real, dim, keepdim, divide_by_n)[0]
                + l1norm_imag.forward(diff.imag, dim, keepdim, divide_by_n)[0],
            )
        else:
            l1norm = L1Norm(weight=self.weight, target=self.target, divide_by_n=divide_by_n)
            return (l1norm.forward(diff, dim, keepdim, divide_by_n)[0],)

    def prox(
        self, x: torch.Tensor, sigma: torch.Tensor, dim: Sequence[int] | None = None, divide_by_n: bool | None = None
    ) -> tuple[torch.Tensor]:
        """Proximal Mapping of the L1 Norm.

        Compute the proximal mapping of the L1-norm with C identified as R^2.

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
        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n

        if x.is_complex():
            l1norm_real = L1Norm(weight=self.weight, target=self.target.real, divide_by_n=divide_by_n)
            l1norm_imag = L1Norm(weight=self.weight, target=self.target.imag, divide_by_n=divide_by_n)
            return (
                torch.complex(
                    l1norm_real.prox(x.real, sigma=sigma, dim=dim, divide_by_n=divide_by_n)[0],
                    l1norm_imag.prox(x.imag, sigma=sigma, dim=dim, divide_by_n=divide_by_n)[0],
                ),
            )
        else:
            l1norm = L1Norm(weight=self.weight, target=self.target, divide_by_n=divide_by_n)
            return ((l1norm.prox(x, sigma=sigma, dim=dim, divide_by_n=divide_by_n)[0]),)

    def prox_convex_conj(
        self, x: torch.Tensor, sigma: torch.Tensor, dim: Sequence[int] | None = None, divide_by_n: bool | None = None
    ) -> tuple[torch.Tensor]:
        """Convex conjugate of the L1 Norm with C identified as R^2.

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
            Proximal of the convex conjugate applied to the input tensor
        """
        if dim is None:
            dim = self.dim
        if divide_by_n is None:
            divide_by_n = self.divide_by_n

        if x.is_complex():
            l1norm_real = L1Norm(weight=self.weight, target=self.target.real, divide_by_n=divide_by_n)
            l1norm_imag = L1Norm(weight=self.weight, target=self.target.imag, divide_by_n=divide_by_n)
            return (
                torch.complex(
                    l1norm_real.prox_convex_conj(x.real, sigma=sigma, dim=dim, divide_by_n=divide_by_n)[0],
                    l1norm_imag.prox_convex_conj(x.imag, sigma=sigma, dim=dim, divide_by_n=divide_by_n)[0],
                ),
            )
        else:
            l1norm = L1Norm(weight=self.weight, target=self.target, divide_by_n=divide_by_n)
            return (l1norm.prox_convex_conj(x, sigma=sigma, dim=dim, divide_by_n=divide_by_n)[0],)
