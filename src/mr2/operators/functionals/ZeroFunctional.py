"""Zero functional."""

from collections.abc import Sequence

import torch

from mr2.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex


class ZeroFunctional(ElementaryProximableFunctional):
    """The constant zero functional."""

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Evaluate the zero functional for the given input tensor.

        This functional always returns a tensor of zeros.


        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Tensor of zeros. The shape is determined by the input `x` and the `dim` and `keepdim` at initialization.
            If `dim` is `None`, the shape matches `x`. Else dimensions of `x` indexed by `dim` are reduced to 1 if
            `keepdim` is `True`, otherwise they are removed.

        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of ZeroFunctional.

        .. note::
            Prefer calling the instance of the ZeroFunctional as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
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
        throw_if_negative_or_complex(sigma)
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.weight.dtype), self.target.dtype)
        return (x.to(dtype=dtype),)

    def prox_convex_conj(self, x: torch.Tensor, sigma: float | torch.Tensor = 1.0) -> tuple[torch.Tensor,]:
        r"""Apply the proximal operator of the convex conjugate of the functional to a tensor.

        The convex conjugate of the zero functional is the indicator function over :math:`C^N \setminus \{0\}`,
        which evaluates to infinity for all values of `x` except zero.
        If ``sigma > 0``, the proximal operator of the scaled convex conjugate is constant zero,
        otherwise it is the identity.

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
        throw_if_negative_or_complex(sigma)
        sigma = torch.as_tensor(sigma)
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.weight.dtype), self.target.dtype)
        result = torch.where(sigma == 0, x, torch.zeros_like(x)).to(dtype=dtype)
        return (result,)
