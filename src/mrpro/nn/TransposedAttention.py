"""Transposed Attention from Restormer."""

import torch
from einops import rearrange
from torch.nn import Module, Parameter

from mrpro.nn.ndmodules import ConvND


class TransposedAttention(Module):
    """Transposed Self Attention from Restormer.

    Implements the transposed self-attention, i.e. channel-wise multihead self-attention,
    layer from Restormer [ZAM22]_.

    References
    ----------
    .. [ZAM22] Zamir, Syed Waqas, et al. "Restormer: Efficient transformer for high-resolution image restoration."
       CVPR 2022, https://arxiv.org/pdf/2111.09881.pdf
    """

    def __init__(self, dim: int, channels_in: int, channels_out: int, n_heads: int):
        """Initialize a TransposedAttention layer.

        Parameters
        ----------
        dim
            input dimension
        channels_in
            Number of channels in the input tensor.
        channels_out
            Number of channels in the output tensor.
        n_heads
            Number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        self.temperature = Parameter(torch.ones(n_heads, 1, 1))
        channels_per_head = channels_in // n_heads
        self.to_qkv = ConvND(dim)(channels_in, channels_per_head * n_heads * 3, kernel_size=1)
        self.qkv_dwconv = ConvND(dim)(
            channels_per_head * n_heads * 3,
            channels_per_head * n_heads * 3,
            kernel_size=3,
            groups=channels_in * 3,
            padding=1,
            bias=False,
        )
        self.to_out = ConvND(dim)(channels_per_head * n_heads, channels_out, kernel_size=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transposed attention.

        Parameters
        ----------
        x
            The input tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transposed attention."""
        qkv = self.qkv_dwconv(self.to_qkv(x))
        q, k, v = rearrange(qkv, 'b (qkv heads channels) ... -> qkv b heads (...) channels', heads=self.n_heads, qkv=3)
        q = torch.nn.functional.normalize(q, dim=-1) * self.temperature
        k = torch.nn.functional.normalize(k, dim=-1)
        attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = rearrange(attention, '... heads points channels -> ... (heads channels) points').unflatten(
            -1, x.shape[2:]
        )
        out = self.to_out(out)
        return out
