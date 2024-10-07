"""Zero functional."""

import torch

from mrpro.operators import ElementaryProximableFunctional


class ZeroFunctional(ElementaryProximableFunctional):
    """The constant zero functional."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the functional to the tensor.

        Always returns 0.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
        Result of the functional applied to x.
        """
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.weight.dtype), self.target.dtype)
        return (torch.zeros_like(x, dtype=dtype).sum(dim=self.dim, keepdim=self.keepdim),)

    def prox(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:
        """Apply the proximal operator to a tensor.

        Always returns x.

        Parameters
        ----------
        x
            input tensor
        sigma
            step size

        Returns
        -------
            Result of the proximal operator applied to x
        """
        self._throw_if_negative_or_complex(sigma)
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.weight.dtype), self.target.dtype)
        return (x.to(dtype=dtype),)

    def prox_convex_conj(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:
        """Apply the proximal operator of the convex conjugate of the functional to a tensor.

        Always returns x.

        Parameters
        ----------
        x
            input tensor
        sigma
            step size

        Returns
        -------
            Result of the proximal operator of the convex conjugate applied to x
        """
        self._throw_if_negative_or_complex(sigma)
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.weight.dtype), self.target.dtype)
        return (torch.zeros_like(x, dtype=dtype),)
