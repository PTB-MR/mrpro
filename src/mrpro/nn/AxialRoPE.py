"""Rotary Position Embedding (RoPE)."""

from collections.abc import Sequence

import torch
from einops import rearrange
from torch.nn import Module


@torch.compile
def get_theta(shape: Sequence[int], n_embedding_channels: int, device: torch.device) -> torch.Tensor:
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
    return rearrange(freqs * position[..., None], '... dim freqs ->... 1 (dim freqs)')


class AxialRoPE(Module):
    """Axial Rotary Position Embedding.

    Applies rotary position embeddings along each axis independently.
    """

    freqs: torch.Tensor

    def __init__(
        self,
        n_heads: int,
        non_embed_fraction: float = 0.0,
    ):
        """Initialize AxialRoPE.

        Parameters
        ----------
        n_heads
            Number of attention heads
        non_embed_fraction
            Fraction of channels not used for embedding
        """
        super().__init__()
        self.non_embed_fraction = non_embed_fraction
        if non_embed_fraction < 0 or non_embed_fraction > 1:
            raise ValueError('non_embed_fraction must be between 0 and 1')
        self.n_heads = n_heads

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply rotary embeddings to input tensors.

        Parameters
        ----------
        *tensors
            Tensors to apply rotary embeddings to
        """
        if self.non_embed_fraction == 1.0:
            return tensors

        shape = tensors[0].shape
        if not all(t.shape == shape for t in tensors):
            raise ValueError('All tensors must have the same shape')
        device = tensors[0].device
        if not all(t.device == device for t in tensors):
            raise ValueError('All tensors must be on the same device')

        shape, n_channels = shape[1:-1], shape[-1]
        if n_channels % self.n_heads:
            raise ValueError(f'Number of channels {n_channels} must be divisible by number of heads {self.n_heads}')
        n_channels_per_head = n_channels // self.n_heads
        tensors = tuple(t.unflatten(-1, (self.n_heads, -1)) for t in tensors)
        n_embedding_channels = int(n_channels_per_head * (1 - self.non_embed_fraction))
        theta = get_theta(shape, n_embedding_channels, device)
        return tuple(self.apply_rotary_emb(t, theta).flatten(-2) for t in tensors)

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
