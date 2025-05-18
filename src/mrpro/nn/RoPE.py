from math import log

import torch


# Rotary position embeddings
@torch.compile
def apply_rotary_emb_(x: torch.Tensor, theta: torch.Tensor, conjugated: bool):
    """Adds the rotary embedding to the input tensor (inplace).

    This is a helper function for the `AxialRoPE` class.
    """
    n_emb = theta.shape[-1] * 2
    if n_emb > x.shape[-1]:
        raise ValueError('More theta values then channels//2 in the input tensor.')
    x1, x2 = x[..., :n_emb].chunk(2, dim=-1)
    dtype = torch.promote_type(torch.result_type(x, theta), torch.float32)
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conjugated else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class RotaryEmbedding_(torch.autograd.Function):
    """Adds the rotary embedding to the input tensor (inplace).

    This is a autograd helper class for the `AxialRoPE` class.
    """

    @staticmethod
    def forward(x: torch.Tensor, theta: torch.Tensor, conjugated: bool) -> torch.Tensor:
        apply_rotary_emb_(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs: tuple[torch.Tensor, torch.Tensor, bool], output: torch.Tensor):
        _, theta, conjugated = inputs
        ctx.save_for_backward(theta)
        ctx.conjugated = conjugated

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (theta,) = ctx.saved_tensors
        apply_rotary_emb_(grad_output, theta, conjugated=not ctx.conjugated)
        return grad_output, None, None


class AxialRoPE(Module):
    def __init__(self, dim: int, d_head: int, n_heads: int, headpos: int = -2, non_embed_fraction: float = 0.5):
        super().__init__()
        log_min = log(torch.pi)
        log_max = log(100 ** (1 / dim) * torch.pi)
        d_per_head = int(d_head / dim * (1 - non_embed_fraction))
        freqs = torch.linspace(log_min, log_max, n_heads * d_per_head).exp()
        freqs = freqs.view(-1, n_heads).T
        freqs = freqs.unsqueeze(-2).repeat(1, dim, 1).contiguous()
        self.freqs = torch.nn.Parameter(freqs)
        self.headpos = headpos

    def get_theta(self, pos):
        return (self.freqs * pos[..., None, :, None]).flatten(start_dim=-2).movedim(-2, self.headpos)

    def forward(self, pos, *tensors):
        theta = self.get_theta(pos)
        tuple(RotaryEmbedding_.apply(x, theta, False) for x in tensors)

    @staticmethod
    def make_axial_positions(*shape):
        shape = torch.as_tensor(shape)
        m = shape.max()
        pos = torch.stack(
            torch.meshgrid([torch.linspace(-1 + 1 / s, 1 - 1 / s, s) * (s / m) for s in shape], indexing='ij'), -1
        )
        return pos
