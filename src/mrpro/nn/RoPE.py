"""Rotary Position Embedding (RoPE)."""

import torch
from torch.nn import Module


@torch.compile
def apply_rotary_emb_(x: torch.Tensor, theta: torch.Tensor, conjugated: bool) -> None:
    """Add rotary embedding to the input tensor (inplace).

    This is a helper function for the `AxialRoPE` class.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to modify
    theta : torch.Tensor
        Rotation angles
    conjugated : bool
        Whether to use conjugated rotation
    """
    n_emb = theta.shape[-1] * 2
    if n_emb > x.shape[-1]:
        raise ValueError(f'Embedding dimension {n_emb} is larger than input dimension {x.shape[-1]}')
    x1, x2 = x[..., :n_emb].chunk(2, dim=-1)
    if conjugated:
        x1, x2 = x2, x1
    x[..., :n_emb] = torch.cat([x1 * theta.cos() - x2 * theta.sin(), x2 * theta.cos() + x1 * theta.sin()], dim=-1)


class RotaryEmbedding(torch.autograd.Function):
    """Custom autograd function for rotary embeddings."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        theta: torch.Tensor,
        conjugated: bool,
    ) -> torch.Tensor:
        """Apply rotary embedding in forward pass."""
        apply_rotary_emb_(x, theta, conjugated)
        return x

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: tuple[torch.Tensor, torch.Tensor, bool], _output: torch.Tensor
    ) -> None:
        """Save tensors for backward pass."""
        _, theta, conjugated = inputs
        ctx.save_for_backward(theta)
        ctx.conjugated = conjugated  # type: ignore[attr-defined]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """Apply backward pass."""
        (theta,) = ctx.saved_tensors  # type: ignore[attr-defined]
        apply_rotary_emb_(grad_output, theta, ctx.conjugated)  # type: ignore[attr-defined]
        return grad_output, None, None


class AxialRoPE(Module):
    """Axial Rotary Position Embedding.

    Applies rotary position embeddings along each axis independently.
    """

    freqs: torch.Tensor

    def __init__(
        self,
        n_dim: int,
        n_channels: int,
        n_heads: int,
        channels_last: bool = True,
        non_embed_fraction: float = 0.5,
    ):
        """Initialize AxialRoPE.

        Parameters
        ----------
        n_dim
            Number of (spatial-like) dimensions of the input
        n_channels
            Number of channels
        n_heads
            Number of attention heads
        channels_last
            Whether the channels are the last dimension or dimension 1.
        non_embed_fraction
            Fraction of channels not used for embedding
        """
        super().__init__()
        log_min = torch.log(torch.tensor(torch.pi))
        log_max = torch.log(torch.tensor(10000.0))
        if n_channels % n_heads:
            raise ValueError(f'Number of channels {n_channels} must be divisible by number of heads {n_heads}')
        channels_per_head = n_channels // n_heads
        freqs = torch.exp(torch.linspace(log_min, log_max, channels_per_head // 2))
        self.register_buffer('freqs', freqs)
        self.channels_last = channels_last
        self.n_heads = n_heads

    def get_theta(self, pos: torch.Tensor) -> torch.Tensor:
        """Get rotation angles for given positions.

        Parameters
        ----------
        pos
            Position tensor

        Returns
        -------
            Rotation angles
        """
        return (self.freqs * pos[..., None, :, None]).flatten(start_dim=-2)

    def forward(self, pos: torch.Tensor, *tensors: torch.Tensor) -> None:
        """Apply rotary embeddings to input tensors.

        Parameters
        ----------
        pos
            Position tensor
        *tensors : torch.Tensor
            Tensors to apply rotary embeddings to
        """
        theta = self.get_theta(pos)
        if not self.channels_last:
            tensors = tuple(t.movedim(-1, 1) for t in tensors)
        tuple(RotaryEmbedding.apply(x, theta, False) for x in tensors)

    @staticmethod
    def make_axial_positions(*shape: int) -> torch.Tensor:
        """Create axial position tensors.

        Parameters
        ----------
        *shape : int
            Shape of the position tensor

        Returns
        -------
        torch.Tensor
            Position tensor
        """
        m = torch.as_tensor(shape).max()
        pos = torch.stack(
            [torch.arange(s, device=m.device) - s // 2 for s in shape],
            dim=-1,
        )
        return pos
