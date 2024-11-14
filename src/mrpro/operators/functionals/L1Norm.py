"""L1 Norm."""

import torch

from mrpro.operators.Functional import ElementaryProximableFunctional


class L1Norm(ElementaryProximableFunctional):
    r"""Functional class for the L1 Norm.

    This implements the functional given by
    :math:`f: C^N -> [0, \infty), x ->  \| W (x-b)\|_1`,
    where W is a either a scalar or tensor that corresponds to a (block-) diagonal operator
    that is applied to the input.

    In most cases, consider setting divide_by_n to true to be independent of input size.

    The norm of the vector is computed along the dimensions given at initialization.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Forward method.

        Compute the l1-norm of the input.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            l1 norm of the input tensor
        """
        value = (self.weight * (x - self.target)).abs()

        if self.divide_by_n:
            return (torch.mean(value, dim=self.dim, keepdim=self.keepdim),)
        else:
            return (torch.sum(value, dim=self.dim, keepdim=self.keepdim),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal Mapping of the L1 Norm.

        Compute the proximal mapping of the L1-norm.

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
        self._throw_if_negative_or_complex(sigma)
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

        Compute the proximal mapping of the convex conjugate of the L1-norm.

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
        self._throw_if_negative_or_complex(sigma)
        diff = x - sigma * self.target
        threshold = self._divide_by_n(self.weight.abs(), torch.broadcast_shapes(x.shape, self.weight.shape))
        x_out = torch.sgn(diff) * torch.clamp_max(diff.abs(), threshold.abs())
        x_out = x_out.to(torch.result_type(threshold, x_out))
        return (x_out,)
