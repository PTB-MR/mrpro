"""L2 Squared Norm."""

import torch

from mr2.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex


class L2NormSquared(ElementaryProximableFunctional):
    r"""Functional class for the squared L2 Norm.

    This implements the functional given by
    :math:`f: C^N \rightarrow [0, \infty), x \rightarrow \| W (x-b)\|_2^2`,
    where :math:`W` is either a scalar or tensor that corresponds to a (block-) diagonal operator
    that is applied to the input. This is, for example, useful for non-Cartesian MRI
    reconstruction when using a density-compensation function for k-space pre-conditioning,
    for masking of image data, or for spatially varying regularization weights.

    In most cases, consider setting `divide_by_n` to `True` to be independent of input size.
    Alternatively, the functional :class:`mr2.operators.functionals.MSE` can be used.
    The norm is computed along the dimensions given at initialization, all other dimensions are
    considered batch dimensions.
    """

    def __call__(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        r"""Compute the squared L2 norm of the input tensor.

        Calculates :math:`\| W * (x - b) \|_2^2`, where :math:`W` is `weight` and :math`b` is `target`.
        The squared norm is computed along dimensions specified by `dim`.
        If `divide_by_n` is `True`, the result is averaged over these
        dimensions; otherwise, it's summed.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The squared L2 norm. If `keepdim` is `True`, the dimensions `dim` are retained
            with size 1; otherwise, they are reduced.
        """
        return super().__call__(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Apply forward of L2NormSquared.

        .. note::
            Prefer calling the instance of the L2NormSquared as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
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

        Apply the proximal mapping of the squared L2 norm.

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

        Apply the proximal mapping of the convex conjugate of the squared L2 norm.

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
        throw_if_negative_or_complex(sigma)
        weight_square = self._divide_by_n(
            self.weight.conj() * self.weight, torch.broadcast_shapes(x.shape, self.target.shape, self.weight.shape)
        )

        x_out = (2 * weight_square * (x - sigma * self.target)) / (sigma + 2 * weight_square)
        return (x_out,)
