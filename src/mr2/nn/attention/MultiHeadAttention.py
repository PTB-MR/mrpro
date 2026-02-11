"""Multi-head Attention."""

import torch
from einops import rearrange
from torch.nn import Linear, Module

from mr2.nn.AxialRoPE import AxialRoPE


class MultiHeadAttention(Module):
    """Multi-head Attention.

    Implements multihead scaled dot-product attention and supports "image-like" inputs,
    i.e. `batch, channels, *spatial_dims` as well as "transformer-like" inputs, `batch, sequence, features`.
    """

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_heads: int,
        features_last: bool = False,
        p_dropout: float = 0.0,
        n_channels_cross: int | None = None,
        rope_embed_fraction: float = 0.0,
    ):
        """Initialize the Multi-head Attention.

        Parameters
        ----------
        n_channels_in
            Number of input channels.
        n_channels_out
            Number of output channels.
        n_heads
            number of attention heads
        features_last
            Whether the features dimension is the last dimension, as common in transformer models,
            or the second dimension, as common in image models.
        p_dropout
            Dropout probability.
        n_channels_cross
            Number of channels for cross-attention. If `None`, use `n_channels_in`.
        rope_embed_fraction
            Fraction of channels to embed with RoPE.
        """
        super().__init__()
        n_channels_kv = n_channels_cross if n_channels_cross is not None else n_channels_in
        channels_per_head_q = n_channels_in // n_heads
        channels_per_head_kv = n_channels_kv // n_heads
        self.to_q = Linear(n_channels_in, channels_per_head_q * n_heads)
        self.to_kv = Linear(n_channels_kv, channels_per_head_kv * n_heads * 2)
        self.p_dropout = p_dropout
        self.features_last = features_last
        self.to_out = Linear(n_channels_in, n_channels_out)
        self.n_heads = n_heads
        self.rope = AxialRoPE(rope_embed_fraction)

    def __call__(self, x: torch.Tensor, cross_attention: torch.Tensor | None = None) -> torch.Tensor:
        """Apply multi-head attention.

        Parameters
        ----------
        x
            The input tensor.
        cross_attention
            The key and value tensors for cross-attention. If `None`, self-attention is applied.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, cross_attention)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        if not self.features_last:
            x = x.moveaxis(1, -1)
        return x.flatten(1, -2)

    def forward(self, x: torch.Tensor, cross_attention: torch.Tensor | None = None) -> torch.Tensor:
        """Apply multi-head attention."""
        if cross_attention is None:
            cross_attention = x
        if not self.features_last:
            x = x.moveaxis(1, -1)
            cross_attention = cross_attention.moveaxis(1, -1)

        query = rearrange(self.to_q(x), 'batch ... (heads channels) -> batch heads ... channels ', heads=self.n_heads)
        key, value = rearrange(
            self.to_kv(cross_attention),
            'batch ... (kv heads channels) -> kv batch heads ... channels ',
            heads=self.n_heads,
            kv=2,
        )
        query, key = self.rope(query, key)  # NO-OP if rope_embed_fraction is 0.0
        query, key, value = query.flatten(2, -2), key.flatten(2, -2), value.flatten(2, -2)
        y = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=self.p_dropout, is_causal=False
        )
        y = rearrange(y, '... heads L channels -> ... L (heads channels)')
        out = self.to_out(y).reshape(x.shape)

        if not self.features_last:
            out = out.moveaxis(-1, 1)

        return out
