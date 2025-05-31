"""L1 Norm with :math:`C` as :math:`R^2`."""

import torch

from mrpro.operators.Functional import ElementaryProximableFunctional


class L1NormViewAsReal(ElementaryProximableFunctional):
    r"""Functional class for the L1 Norm, where C is identified with R^2.

    This implements the functional given by
    :math:`f: C^N -> [0, \infty), x ->  \|W_r * Re(x-b) )\|_1 + \|( W_i * Im(x-b) )\|_1`,
    where :math:`W_r` and :math:`W_i` are a either scalars or tensors and `*` denotes element-wise multiplication.

    If the parameter `weight` is real-valued, :math:`W_r` and :math:`W_i` are both set to `weight`.
    If it is complex-valued, :math:`W_r` and :math:`W_I` are set to the real and imaginary part, respectively.

    In most cases, consider setting `divide_by_n` to `true` to be independent of input size.

    The norm of the vector is computed along the dimensions set at initialization.
    """

    def __call__(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Compute the L1 norm, viewing complex numbers as R^2.

        Calculates :math:`\|W_r * Re(x-b)\|_1 + \|W_i * Im(x-b)\|_1`.
        If `self.weight` is real, :math:`W_r = W_i = self.weight`.
        If `self.weight` is complex, :math:`W_r = Re(self.weight)` and :math:`W_i = Im(self.weight)`.
        `b` is `self.target`. The norm is computed along `self.dim`.
        If `self.divide_by_n` is true, the result is averaged; otherwise, summed.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor]
            A tuple containing a single tensor representing the L1 norm (viewed as R^2).
            If `self.keepdim` is true, `self.dim` are retained; otherwise, reduced.
        """
        return super().__call__(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Apply forward of L1NormViewAsReal.

        Note: Do not use. Instead, call the instance of the Operator as operator(x)"""
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

        Apply the proximal mapping of the L1 norm with :math:`C` identified as :math:`R^2`.

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
