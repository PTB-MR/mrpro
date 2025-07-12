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

    def __init__(self, dim: int, d_head: int, n_heads: int, headpos: int = -2, non_embed_fraction: float = 0.5):
        """Initialize AxialRoPE.

        Parameters
        ----------
        dim : int
            Dimension of the input space
        d_head : int
            Dimension of each attention head
        n_heads : int
            Number of attention heads
        headpos : int, optional
            Position of the head dimension
        non_embed_fraction : float, optional
            Fraction of dimensions to not embed
        """
        super().__init__()
        log_min = torch.log(torch.tensor(torch.pi))
        log_max = torch.log(torch.tensor(10000.0))
        freqs = torch.exp(torch.linspace(log_min, log_max, d_head // 2))
        self.register_buffer('freqs', freqs)
        self.headpos = headpos

    def get_theta(self, pos: torch.Tensor) -> torch.Tensor:
        """Get rotation angles for given positions.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor

        Returns
        -------
        torch.Tensor
            Rotation angles
        """
        return (self.freqs * pos[..., None, :, None]).flatten(start_dim=-2).movedim(-2, self.headpos)

    def forward(self, pos: torch.Tensor, *tensors: torch.Tensor) -> None:
        """Apply rotary embeddings to input tensors.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor
        *tensors : torch.Tensor
            Tensors to apply rotary embeddings to
        """
        theta = self.get_theta(pos)
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
