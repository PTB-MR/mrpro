"""Feature-wise Linear Modulation."""

import torch
from torch.nn import Identity, Linear, Module, Sequential, SiLU

from mrpro.nn.EmbMixin import EmbMixin
from mrpro.utils.reshape import unsqueeze_tensors_right


class FiLM(EmbMixin, Module):
    """Feature-wise Linear Modulation.

    Feature-wise Linear Modulation from [FiLM]_

    References
    ----------
    ..[FiLM] Perez, L., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. "Film: Visual reasoning with a general conditioning layer." AAAI (2018).
      https://arxiv.org/abs/1709.07871
    """

    def __init__(self, channels: int, channels_emb: int) -> None:
        """Initialize FiLM.

        Parameters
        ----------
        channels
            The number of channels in the input tensor.
        channels_emb
            The number of channels in the embedding tensor.
        """
        super().__init__()
        if channels_emb > 0:
            self.project = Sequential(
                SiLU(),
                Linear(channels_emb, 2 * channels),
            )
        else:
            self.project = Identity()

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FiLM.

        Parameters
        ----------
        x
            The input tensor.
        emb
            The embedding tensor.
        """
        return super().__call__(x, emb)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """Apply FiLM."""
        if emb is None:
            return x

        emb = self.project(emb)
        scale, shift = emb.chunk(2, dim=1)
        scale, shift = unsqueeze_tensors_right(scale, shift, ndim=x.ndim)
        return x * (1 + scale) + shift
