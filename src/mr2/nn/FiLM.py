"""Feature-wise Linear Modulation."""

import torch
from torch.nn import Linear, Module

from mr2.nn.CondMixin import CondMixin
from mr2.utils.reshape import unsqueeze_tensors_right


class FiLM(CondMixin, Module):
    """Feature-wise Linear Modulation.

    Feature-wise Linear Modulation from [FiLM]_ to condition a network on a conditioning tensor.


    References
    ----------
    ..[FiLM] Perez, L., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. "FiLM : Visual reasoning with a general
      conditioning layer." AAAI (2018). https://arxiv.org/abs/1709.07871
    """

    features_last: bool

    def __init__(self, channels: int, cond_dim: int, features_last: bool = False) -> None:
        """Initialize FiLM.

        Parameters
        ----------
        channels
            The number of channels in the input tensor.
        cond_dim
            The dimension of the conditioning tensor.
        features_last
            Whether the features are in the last dimension of the input tensor (e.g. transformer tokens)
            or in the second dimension (e.g. image tensors).
        """
        super().__init__()
        self.project = Linear(cond_dim, 2 * channels) if cond_dim > 0 else None
        self.features_last = features_last

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FiLM.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The conditioning tensor.
        """
        return super().__call__(x, cond=cond)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FiLM."""
        if cond is None or self.project is None:
            return x

        if self.features_last:
            x = x.moveaxis(-1, 1)

        scale, shift = self.project(cond).chunk(2, dim=1)
        scale, shift = unsqueeze_tensors_right(scale, shift, ndim=x.ndim)
        x = x * (1 + scale) + shift

        if self.features_last:
            x = x.moveaxis(1, -1)

        return x
