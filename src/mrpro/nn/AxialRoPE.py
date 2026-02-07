"""Rotary Position Embedding (RoPE)."""

from collections.abc import Sequence

import torch
from einops import rearrange
from torch.nn import Module


@torch.compile
def get_theta(
    shape: Sequence[int], n_embedding_channels: int, device: torch.device
) -> torch.Tensor:  # pragma: no cover
    """Get rotation angles.

    Parameters
    ----------
    shape
        Spatial shape of the input tensor to use for the position embedding,
        i.e. the shape excluding batch and channel dimensions.
    n_embedding_channels
        Number of embedding channels per head
    device
        Device to create the rotation angles on

    Returns
    -------
        Rotation angles
    """
    position = torch.stack(
        torch.meshgrid([torch.arange(s, device=device) - s // 2 for s in shape], indexing='ij'), dim=-1
    )
    log_min = torch.log(torch.tensor(torch.pi))
    log_max = torch.log(torch.tensor(10000.0))
    freqs = torch.exp(torch.linspace(log_min, log_max, n_embedding_channels // (2 * position.shape[-1]), device=device))
    return rearrange(freqs * position[..., None], '... dim freqs ->... (dim freqs)')


class AxialRoPE(Module):
    """Axial Rotary Position Embedding.

    Applies rotary position embeddings along each axis independently.
    """

    embed_fraction: float
    freqs: torch.Tensor  # explicit annotation kept for static type checking

    def __init__(
        self,
        embed_fraction: float = 1.0,
    ):
        """Initialize AxialRoPE.

        Parameters
        ----------
        embed_fraction
            Fraction of channels used for embedding
        """
        super().__init__()
        self.embed_fraction: float = float(embed_fraction)
        if embed_fraction < 0 or embed_fraction > 1:
            raise ValueError('embed_fraction must be between 0 and 1')

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply rotary embeddings to input tensors.

        Parameters
        ----------
        *tensors
            Tensors to apply rotary embeddings to.
            Shape must be `(batch, heads, *spatial_dims, channels)`.
        """
        if self.embed_fraction == 0.0:
            return tensors

        shape = tensors[0].shape
        if not all(t.shape == shape for t in tensors):
            raise ValueError('All tensors must have the same shape')
        device = tensors[0].device
        if not all(t.device == device for t in tensors):
            raise ValueError('All tensors must be on the same device')

        shape, n_channels_per_head = shape[2:-1], shape[-1]
        n_embedding_channels = int(n_channels_per_head * self.embed_fraction)
        theta = get_theta(shape, n_embedding_channels, device)
        return tuple(self.apply_rotary_emb(t, theta) for t in tensors)

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Add rotary embedding to the input tensor.

        Parameters
        ----------
        x
            Input tensor to modify
        theta
            Rotation angles
        """
        n_emb = theta.shape[-1] * 2
        if n_emb > x.shape[-1]:
            raise ValueError(f'Embedding dimension {n_emb} is larger than input dimension {x.shape[-1]}')
        (x1, x2), x_unembed = x[..., :n_emb].chunk(2, dim=-1), x[..., n_emb:]
        result = torch.cat(
            [x1 * theta.cos() - x2 * theta.sin(), x2 * theta.cos() + x1 * theta.sin(), x_unembed], dim=-1
        )
        return result
