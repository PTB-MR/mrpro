"""L1 Norm."""

import torch

from mrpro.operators.Functional import ProximableFunctional


class L1NormViewAsReal(ProximableFunctional):
    r"""Functional class for the L1 Norm, where C is identified with R^2.

    This implements the functional given by
        f: C^N --> [0, \infty), x ->  \|Wr.*Re(x-b) )\|_1 + \|( Wi.*Im(x-b) )\|_1
    where Wr and Wi are a either scalars or tensors and .* denotes element-wise multiplication.

    If the weight parameter W is real-valued, Wr and Wi are both set to W.
    If it is complex-valued, Wr and Wi are set to the real and imaginary part of W, respectively.

    The norm of the vector is computed along the dimensions given by "dim".
    Further, it is possible to scale the functional by N, i.e. by the number voxels of
    the elements of the vector space that is spanned by the dimensions indexed by "dim".
    If "dim" is set to None and "keepdim" to False, the result is a single number, which
    is typically of interest for computing loss functions.

    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Forward method.

        Compute the L1-norm of the input with C identified as R^2.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            L1 norm of the input tensor, where C is identified as R^2
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target
        if diff.is_complex() and self.weight.is_complex():
            value = (self.weight.real * diff.real).abs() + (self.weight.imag * diff.imag).abs()
        elif diff.is_complex():
            value = (self.weight * diff.real).abs() + (self.weight * diff.imag).abs()
        else:
            value = (self.weight * diff).abs()

        if self.divide_by_n:
            return (torch.mean(value, dim=self.dim, keepdim=self.keepdim),)
        else:
            return (torch.sum(value, dim=self.dim, keepdim=self.keepdim),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal Mapping of the L1 Norm.

        Compute the proximal mapping of the L1-norm with C identified as R^2.

        Parameters
        ----------
        x
            input tensor
        sigma
            real valued scaling factor

        Returns
        -------
            Proximal mapping applied to the input tensor
        """
        diff = x - self.target
        threshold = self._divide_by_n(self.weight * sigma, torch.broadcast_shapes(x.shape, self.weight.shape))
        out = torch.sgn(diff.real) * torch.clamp_max(diff.real.abs(), threshold.real.abs())
        if diff.is_complex():
            threshold_imag = threshold.imag if self.weight.is_complex() else threshold
            imag = torch.sgn(diff.imag) * torch.clamp_max(diff.imag.abs(), threshold_imag.abs())
            out = torch.complex(out, imag)
        out = out.to(torch.result_type(threshold, out))
        return (out,)
