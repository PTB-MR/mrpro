"""L1 Norm with :math:`C` as :math:`R^2`."""

import torch

from mr2.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex


class L1NormViewAsReal(ElementaryProximableFunctional):
    r"""Functional class for the L1 Norm, where C is identified with :math:`R^2`.

    This implements the functional given by
    :math:`f: C^N \rightarrow [0, \infty), x \rightarrow \|W_r * \mathrm{Re}(x-b))\|_1 +\|( W_i *\mathrm{Im}(x-b))\|_1`,
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
        r"""Compute the L1 norm, viewing complex numbers as R^2.

        Calculates :math:`\|W_r * \mathrm{Re}(x-b)\|_1 + \|W_i * \mathrm{Im}(x-b)\|_1`.
        If `weight` is real, :math:`W_r = W_i = \mathrm{weight}`.
        If `weight` is complex, :math:`W_r = \mathrm{Re}(\mathrm{weight})` and :math:`W_i=\mathrm{Im}(\mathrm{weight})`.
        `b` is `target`. The norm is computed along `dim`.
        If `divide_by_n` is `True`, the result is averaged; otherwise, summed.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The L1 norm. If `keepdim` is `True`, the dimensions `dim` are retained
            with size 1; otherwise, they are reduced.
        """
        return super().__call__(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Apply forward of L1NormViewAsReal.

        .. note::
            Prefer calling the instance of the L1NormViewAsReal operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
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
        throw_if_negative_or_complex(sigma)
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
