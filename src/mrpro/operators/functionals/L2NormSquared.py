"""L2 Squared Norm."""

import torch

from mrpro.operators.Functional import ElementaryProximableFunctional


class L2NormSquared(ElementaryProximableFunctional):
    r"""Functional class for the squared L2 Norm.

    This implements the functional given by
    :math:`f: C^N -> [0, \infty), x -> \| W (x-b)\|_2^2`,
    where :math:`W` is either a scalar or tensor that corresponds to a (block-) diagonal operator
    that is applied to the input. This is, for example, useful for non-Cartesian MRI
    reconstruction when using a density-compensation function for k-space pre-conditioning,
    for masking of image data, or for spatially varying regularization weights.

    In most cases, consider setting divide_by_n to true to be independent of input size.
    Alternatively the functional :class:`mrpro.operators.functionals.MSE` can be used.
    The norm is computed along the dimensions given at initialization, all other dimensions are
    considered batch dimensions.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Forward method.

        Compute the squared L2-norm of the input.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            squared l2 norm of the input tensor
        """
        value = (self.weight * (x - self.target)).abs().square()

        if self.divide_by_n:
            return (torch.mean(value, dim=self.dim, keepdim=self.keepdim),)
        else:
            return (torch.sum(value, dim=self.dim, keepdim=self.keepdim),)

    def prox(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor]:
        """Proximal Mapping of the squared L2 Norm.

        Apply the proximal mapping of the squared L2-norm.

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
        weight_square_2_sigma = self._divide_by_n(
            self.weight.conj() * self.weight * 2 * sigma,
            torch.broadcast_shapes(x.shape, self.target.shape, self.weight.shape),
        )
        x_out = (x + weight_square_2_sigma * self.target) / (1.0 + weight_square_2_sigma)

        return (x_out,)

    def prox_convex_conj(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor]:
        """Convex conjugate of squared L2 Norm.

        Apply the proximal mapping of the convex conjugate of the squared L2-norm.

        Parameters
        ----------
        x
            data tensor
        sigma
            scaling factor

        Returns
        -------
            Proximal of convex conjugate applied to the input tensor
        """
        self._throw_if_negative_or_complex(sigma)
        weight_square = self._divide_by_n(
            self.weight.conj() * self.weight, torch.broadcast_shapes(x.shape, self.target.shape, self.weight.shape)
        )

        x_out = (2 * weight_square * (x - sigma * self.target)) / (sigma + 2 * weight_square)
        return (x_out,)
