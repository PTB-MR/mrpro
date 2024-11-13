"""Zero functional."""

from collections.abc import Sequence

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
        # To ensure that the dtype matches what it would be if we were to apply the weight and target
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.weight.dtype), self.target.dtype).to_real()

        if self.dim is None:
            normal_dim: Sequence[int] = range(x.ndim)
        elif not all(-x.ndim <= d < x.ndim for d in self.dim):
            raise IndexError('Invalid dimension index')
        else:
            normal_dim = [d % x.ndim for d in self.dim] if x.ndim > 0 else []

        if self.keepdim:
            new_shape = [1 if i in normal_dim else s for i, s in enumerate(x.shape)]
        else:
            new_shape = [s for i, s in enumerate(x.shape) if i not in normal_dim]

        return (torch.zeros(new_shape, dtype=dtype, device=self.target.device),)

    def prox(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:
        """Apply the proximal operator to a tensor.

        Always returns x, as the proximal operator of a constant functional is the identity.

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
        r"""Apply the proximal operator of the convex conjugate of the functional to a tensor.

        The convex conjugate of the zero functional is the indicator function over :math:`C^N \setminus {0}`,
        which evaluates to infinity for all values of `x` except zero.
        If sigma>0, the proximal operator of the scaled convex conjugate is constant zero, otherwise it is the identity.

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
        sigma = torch.as_tensor(sigma)
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.weight.dtype), self.target.dtype)
        result = torch.where(sigma == 0, x, torch.zeros_like(x)).to(dtype=dtype)
        return (result,)
