"""L1 Norm."""

import torch

from mrpro.operators.Functional import ProximableFunctional


class L1Norm(ProximableFunctional):
    r"""Functional class for the L1 Norm.

    This implements the functional given by
        :math:`f: C^N --> [0, \infty), x ->  \| W*(x-b)\|_1`,
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
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        value = (self.weight * (x - target)).abs()

        if self.divide_by_n:
            return (torch.mean(value, dim=self.dim, keepdim=self.keepdim),)
        else:
            return (torch.sum(value, dim=self.dim, keepdim=self.keepdim),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float) -> tuple[torch.Tensor]:
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
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - target

        threshold = self.weight * sigma
        threshold = self._divide_by_n(threshold, torch.broadcast_shapes(x.shape, threshold.shape))

        x_out = x - diff / torch.clamp_min((diff / threshold).abs(), 1)
        x_out = x_out.to(torch.result_type(threshold, x_out))
        return (x_out,)

    def prox_convex_conj(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float,
    ) -> tuple[torch.Tensor]:
        """Convex conjugate of the L1 Norm.

        Compute the proximal mapping of the convex conjugate of the L1-norm.

        Parameters
        ----------
            x
                data tensor
            sigma
                scaling factor

        Returns
        -------
            Proximal of the convex conjugate applied to the input tensor
        """
        dtype = torch.promote_types(self.target.dtype, x.dtype)
        x = x.to(dtype)
        target = self.target.to(dtype)
        diff = x - sigma * target
        threshold = self._divide_by_n(self.weight.abs(), torch.broadcast_shapes(x.shape, self.weight.shape))

        if diff.is_complex():
            x_out = torch.polar(torch.clamp(diff.abs(), -threshold, threshold), torch.angle(diff))
        else:
            x_out = torch.clamp(diff, -threshold, threshold)

        return (x_out,)
