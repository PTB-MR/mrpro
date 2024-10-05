"""Zero functional."""

import torch


class ZeroFunctional(ProximableFunctional):
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
        if self.dim is None:
            dim = list(range(x.dim()))
        else:
            dim = [d % x.dim() for d in self.dim]
        if self.keepdim:
            shape = [1 if d in dim else s for d, s in enumerate(x.shape)]
        else:
            shape = [s for d, s in enumerate(x.shape) if d not in dim]
        dtype = torch.promote_types(torch.promote_types(self.target.dtype, self.weight.dtype), x.dtype)
        return (torch.zeros(shape, dtype=dtype, device=self.weight.device),)

    def prox(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:  # noqa ARG002
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
        return (x,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:  # noqa ARG002
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
        return (x,)
