"""Feature-wise Linear Modulation."""

import torch
from torch.nn import Linear, Module

from mrpro.nn.CondMixin import CondMixin
from mrpro.utils.reshape import unsqueeze_tensors_right


class FiLM(CondMixin, Module):
    """Feature-wise Linear Modulation.

    Feature-wise Linear Modulation from [FiLM]_ to condition a network on a conditioning tensor.


    References
    ----------
    ..[FiLM] Perez, L., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. "FiLM : Visual reasoning with a general
      conditioning layer." AAAI (2018). https://arxiv.org/abs/1709.07871
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        """Initialize FiLM.

        Parameters
        ----------
        channels
            The number of channels in the input tensor.
        cond_dim
            The dimension of the conditioning tensor.
        """
        super().__init__()
        self.project = Linear(cond_dim, 2 * channels) if cond_dim > 0 else None

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FiLM.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The conditioning tensor.
        """
        if len(x) != 1:
            raise ValueError('FiLM expects a single input tensor')
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FiLM."""
        if cond is None or self.project is None:
            return x
        scale, shift = self.project(cond).chunk(2, dim=1)

        scale, shift = unsqueeze_tensors_right(scale, shift, ndim=x.ndim)
        return x * (1 + scale) + shift
