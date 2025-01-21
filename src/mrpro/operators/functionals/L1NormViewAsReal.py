"""L1 Norm."""

import torch

from mrpro.operators.Functional import ElementaryProximableFunctional


class L1NormViewAsReal(ElementaryProximableFunctional):
    r"""Functional class for the L1 Norm, where C is identified with R^2.

    This implements the functional given by
    :math:`f: C^N -> [0, \infty), x ->  \|W_r * Re(x-b) )\|_1 + \|( W_i * Im(x-b) )\|_1`
    where :math:`W_r` and :math:`W_i` are a either scalars or tensors and `*` denotes element-wise multiplication.

    If the parameter `weight` is real-valued, :math:`W_r` and :math:`W_i` are both set to `weight`.
    If it is complex-valued, :math:`W_r` and :math:`W_I` are set to the real and imaginary part, respectively.

    In most cases, consider setting divide_by_n to true to be independent of input size.

    The norm of the vector is computed along the dimensions set at initialization.
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

        Apply the proximal mapping of the L1-norm with C identified as R^2.

        Parameters
        ----------
        x
            input tensor
        sigma
            real valued positive scaling factor

        Returns
        -------
            Proximal mapping applied to the input tensor
        """
        self._throw_if_negative_or_complex(sigma)
        diff = x - self.target
        threshold = self._divide_by_n(self.weight * sigma, torch.broadcast_shapes(x.shape, self.weight.shape))
        out = torch.sgn(diff.real) * torch.relu(diff.real.abs() - threshold.real.abs())
        if diff.is_complex():
            threshold_imag = threshold.imag if self.weight.is_complex() else threshold
            imag = torch.sgn(diff.imag) * torch.relu(diff.imag.abs() - threshold_imag.abs())
            out = torch.complex(out, imag)
        out = out + self.target
        out = out.to(torch.result_type(threshold, out))
        return (out,)
