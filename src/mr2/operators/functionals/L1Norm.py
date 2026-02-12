"""L1 Norm."""

import torch

from mr2.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex


class L1Norm(ElementaryProximableFunctional):
    r"""Functional class for the L1 Norm.

    This implements the functional given by
    :math:`f: C^N \rightarrow [0, \infty), x \rightarrow  \| W (x-b)\|_1`,
    where W is a either a scalar or tensor that corresponds to a (block-) diagonal operator
    that is applied to the input.

    In most cases, consider setting `divide_by_n` to `true` to be independent of input size.

    The norm of the vector is computed along the dimensions given at initialization.
    """

    def __call__(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Compute the L1 norm of the input tensor.

        Calculates :math:`|| W * (x - b) ||_1`, where :math:`W` is `weight` and :math:`b` is `target`.
        The norm is computed along dimensions specified by `dim`.
        If `divide_by_n` is true, the result is averaged over these dimensions;
        otherwise, it's summed.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The L1 norm. If `keepdim` is true, the dimensions `dim` are retained
            with size 1; otherwise, they are reduced.
        """
        return super().__call__(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Apply forward of L1Norm.

        .. note::
            Prefer calling the instance of the L1Norm as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        value = (self.weight * (x - self.target)).abs()

        if self.divide_by_n:
            return (torch.mean(value, dim=self.dim, keepdim=self.keepdim),)
        else:
            return (torch.sum(value, dim=self.dim, keepdim=self.keepdim),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal Mapping of the L1 Norm.

        Compute the proximal mapping of the L1 norm.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor

        Returns
        -------
            Proximal mapping applied to the input tensor
        """
        throw_if_negative_or_complex(sigma)
        diff = x - self.target
        threshold = self.weight * sigma
        threshold = self._divide_by_n(threshold, torch.broadcast_shapes(x.shape, threshold.shape))
        x_out = torch.sgn(diff) * torch.relu(diff.abs() - threshold.abs()) + self.target
        x_out = x_out.to(torch.result_type(threshold, x_out))
        return (x_out,)

    def prox_convex_conj(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor]:
        """Convex conjugate of the L1 Norm.

        Compute the proximal mapping of the convex conjugate of the L1 norm.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor

        Returns
        -------
            Proximal of the convex conjugate applied to the input tensor
        """
        throw_if_negative_or_complex(sigma)
        diff = x - sigma * self.target
        threshold = self._divide_by_n(self.weight.abs(), torch.broadcast_shapes(x.shape, self.weight.shape))
        x_out = torch.sgn(diff) * torch.clamp_max(diff.abs(), threshold.abs())
        x_out = x_out.to(torch.result_type(threshold, x_out))
        return (x_out,)
