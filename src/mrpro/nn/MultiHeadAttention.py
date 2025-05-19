"""Multi-head Attention."""

import torch
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
        """
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=channels_in, n_heads=n_heads, batch_first=True, dropout=p_dropout
        )
        self.features_last = features_last
        self.to_out = Linear(channels_in, channels_out)

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
        return x.flatten(2, -2)

    def forward(self, x: torch.Tensor, cross_attention: torch.Tensor | None = None) -> torch.Tensor:
        """Apply multi-head attention."""
        reshaped_x = self._reshape(x)
        reshaped_cross_attention = self._reshape(cross_attention) if cross_attention is not None else reshaped_x

        y = self.mha(reshaped_cross_attention, reshaped_cross_attention, reshaped_x)
        out = self.to_out(y)

        if not self.features_last:
            out = out.moveaxes(-1, 1)

        return out.reshape(x.shape)
