"""Transposed Attention from Restormer."""

import torch
from einops import rearrange
from torch.nn import Module, Parameter

from mr2.nn.ndmodules import convND


class TransposedAttention(Module):
    """Transposed Self Attention from Restormer.

    Implements the transposed self-attention, i.e. channel-wise multihead self-attention,
    layer from Restormer [ZAM22]_.

    References
    ----------
    .. [ZAM22] Zamir, Syed Waqas, et al. "Restormer: Efficient transformer for high-resolution image restoration."
       CVPR 2022, https://arxiv.org/pdf/2111.09881.pdf
    """

    def __init__(self, n_dim: int, n_channels_in: int, n_channels_out: int, n_heads: int):
        """Initialize a TransposedAttention layer.

        Parameters
        ----------
        n_dim
            input dimension
        n_channels_in
            Number of channels in the input tensor.
        n_channels_out
            Number of channels in the output tensor.
        n_heads
            Number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        self.temperature = Parameter(torch.ones(n_heads, 1, 1))
        channels_per_head = n_channels_in // n_heads
        self.to_qkv = convND(n_dim)(n_channels_in, channels_per_head * n_heads * 3, kernel_size=1)
        self.qkv_dwconv = convND(n_dim)(
            channels_per_head * n_heads * 3,
            channels_per_head * n_heads * 3,
            kernel_size=3,
            groups=n_channels_in * 3,
            padding=1,
            bias=False,
        )
        self.to_out = convND(n_dim)(channels_per_head * n_heads, n_channels_out, kernel_size=1)

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
