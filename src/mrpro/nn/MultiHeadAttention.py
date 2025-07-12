"""Multi-head Attention."""

import torch
from einops import rearrange
from torch.nn import Linear, Module


class MultiHeadAttention(Module):
    """Multi-head Attention.

    Implements multihead scaled dot-product attention and supports "image-like" inputs,
    i.e. `batch, channels, *spatial_dims` as well as "transformer-like" inputs, `batch, sequence, features`.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        n_heads: int,
        features_last: bool = False,
        p_dropout: float = 0.0,
        channels_cross: int | None = None,
    ):
        """Initialize the Multi-head Attention.

        Parameters
        ----------
        dim
            Number of spatial dimensions.
        channels_in
            Number of input channels.
        channels_out
            Number of output channels.
        n_heads
            number of attention heads
        features_last
            Whether the features dimension is the last dimension, as common in transformer models,
            or the second dimension, as common in image models.
        p_dropout
            Dropout probability.
        channels_cross
            Number of channels for cross-attention. If `None`, use `channels_in`.
        """
        super().__init__()
        channels_per_head_q = channels_in // n_heads
        channels_per_head_kv = channels_cross // n_heads if channels_cross is not None else channels_in // n_heads
        self.to_q = Linear(channels_in, channels_per_head_q * n_heads)
        self.to_kv = Linear(channels_in, channels_per_head_kv * n_heads * 2)
        self.p_dropout = p_dropout
        self.features_last = features_last
        self.to_out = Linear(channels_in, channels_out)
        self.n_heads = n_heads

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
        reshaped_x = self._reshape(x)
        reshaped_cross_attention = self._reshape(cross_attention) if cross_attention is not None else reshaped_x

        q = rearrange(self.to_q(reshaped_x), '... L (heads dim) -> ... heads L dim ', heads=self.n_heads)
        k, v = rearrange(
            self.to_kv(reshaped_cross_attention),
            '... S (kv heads dim) -> kv ... heads S dim ',
            heads=self.n_heads,
            kv=2,
        )
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.p_dropout, is_causal=False)
        y = rearrange(y, '... heads L dim -> ... L (heads dim)')
        out = self.to_out(y)

        if not self.features_last:
            out = out.moveaxis(-1, 1)

        return out.reshape(x.shape)
