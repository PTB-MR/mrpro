"""Linear self-attention."""

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import Linear, Module, ReLU


class LinearSelfAttention(Module):
    """Linear multi-head self-attention via kernel trick.

    Uses a ReLU kernel to compute attention in O(N) [KAT20]_ time and space.


    References
    ----------
    .. [KAT20] Katharopoulos, Angelos, et al. Transformers are RNNs: Fast autoregressive transformers with linear
       attention. ICML 2020. https://arxiv.org/abs/2006.16236
    """

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_heads: int,
        eps: float = 1e-6,
        features_last: bool = False,
    ):
        """Initialize linear self-attention layer.

        Parameters
        ----------
        n_channels_in
            Input channel dimension.
        n_channels_out
            Output channel dimension.
        n_heads
            Number of attention heads.
        eps
            Small epsilon for numerical stability in normalization.
        features_last
            Whether the channel dimension is the last dimension, as common in transformer models,
            or the second dimension, as common in image models.
        """
        super().__init__()
        self.features_last = features_last
        self.eps = eps
        self.n_heads = n_heads
        channels_per_head = n_channels_in // n_heads
        self.to_qkv = Linear(n_channels_in, 3 * channels_per_head * n_heads)
        self.kernel_function = ReLU()
        self.to_out = Linear(channels_per_head * n_heads, n_channels_out)

    def __call__(self, x: Tensor) -> Tensor:
        """Apply linear self-attention.

        Parameters
        ----------
        x
            Tensor of shape `batch, channels, *spatial_dims` or (`batch, *spatial_dims, channels` if `features_last`)

        Returns
        -------
            Tensor after attention, same shape as input.
        """
        return super().__call__(x)

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear self-attention."""
        orig_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
        if not self.features_last:
            x = x.moveaxis(1, -1)
        spatial_shape = x.shape[1:-1]

        qkv = self.to_qkv(x)
        query, key, value = rearrange(
            qkv, 'batch ... (qkv head channels) -> qkv batch head (...) channels', qkv=3, head=self.n_heads
        )

        query = self.kernel_function(query)
        key = self.kernel_function(key)

        # trick to avoid second attention calculation: add normalization slot
        value = torch.nn.functional.pad(value, (0, 0, 0, 1), mode='constant', value=1.0)

        value_key = value @ key.transpose(-1, -2)
        value_key_query = value_key @ query
        normalization = value_key_query[..., -1:, :] + self.eps
        attn = value_key_query[..., :-1, :] / normalization
        attn = attn.moveaxis(1, -1).flatten(-2)  # join heads and channels
        out = self.to_out(attn)
        out = out.to(orig_dtype)
        out = out.unflatten(-2, spatial_shape)
        if not self.features_last:
            out = out.moveaxis(-1, 1)
        return out
