"""Huber Functional."""

import math

import torch

from mrpro.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex


class Huber(ElementaryProximableFunctional):
    r"""Functional class for the Huber functional.

    This implements the functional given by:
    :math:`f(x) = \sum_i L_\delta(w_i(x_i - b_i))`
    where :math:`L_\delta` is the Huber loss function, :math:`w` is the weight,
    and :math:`b` is the target.

    The Huber function is defined as:
    $$
    L_\delta(a) =
    \begin{cases}
    \frac{1}{2} a^2 & \text{for } |a| \le \delta \\
    \delta (|a| - \frac{1}{2}\delta) & \text{for } |a| > \delta
    \end{cases}
    $$

    In most cases, consider setting `divide_by_n` to `True` to be independent
    of input size.
    """

    def __init__(self, delta: float = 1.0, **kwargs) -> None:
        """Initialize the HuberLoss functional.

        Parameters
        ----------
        delta
            The threshold parameter for the Huber loss. Must be positive.
        **kwargs
            Additional arguments for the ElementaryFunctional base class.
        """
        super().__init__(**kwargs)
        if delta <= 0:
            raise ValueError('delta must be positive')
        self.delta = delta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the Huber loss functional."""
        diff = self.weight * (x - self.target)
        abs_diff = diff.abs()

        l2_loss = 0.5 * diff.square()
        l1_loss = self.delta * (abs_diff - 0.5 * self.delta)

        value = torch.where(abs_diff <= self.delta, l2_loss, l1_loss)

        if self.divide_by_n:
            return (torch.mean(value, dim=self.dim, keepdim=self.keepdim),)
        else:
            return (torch.sum(value, dim=self.dim, keepdim=self.keepdim),)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal operator of the Huber Loss."""
        throw_if_negative_or_complex(sigma)
        sigma_tensor = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        sigma_scaled = self._divide_by_n(sigma_tensor, x.shape)

        w = self.weight.to(x.dtype)
        b = self.target.to(x.dtype)

        diff = x - b
        w_diff = w * diff
        sigma_w_sq = sigma_scaled * w.square()

        condition = w_diff.abs() <= self.delta * (1 + sigma_w_sq)
        prox_l2 = b + diff / (1 + sigma_w_sq)
        threshold = sigma_scaled * w * self.delta
        prox_l1 = x - threshold * torch.sgn(diff)

        x_out = torch.where(condition, prox_l2, prox_l1)
        return (x_out,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal operator of the convex conjugate of the Huber Loss."""
        throw_if_negative_or_complex(sigma)
        sigma_tensor = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)

        w = self.weight.to(x.dtype)
        b = self.target.to(x.dtype)
        w_sq = w.square()

        if self.divide_by_n:
            shape = torch.broadcast_shapes(x.shape, w.shape, b.shape)
            if self.dim is not None:
                dims = [shape[i] for i in self.dim]
            else:
                dims = list(shape)
            N = float(math.prod(dims))
        else:
            N = 1.0

        sigma_eff = sigma_tensor * N
        unconstrained_q = (w_sq / (w_sq + sigma_eff)) * (x - sigma_tensor * b)

        clip_bound = self.delta * w.abs() / N
        q_out = torch.clamp(unconstrained_q, -clip_bound, clip_bound)

        return (q_out,)
