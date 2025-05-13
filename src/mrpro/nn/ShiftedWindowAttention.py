"""Shifted Window Attention."""

import torch
from einops import rearrange
from torch.nn import Module

from mrpro.nn.NDModules import ConvND
from mrpro.utils.reshape import ravel_multi_index
from mrpro.utils.sliding_window import sliding_window


class ShiftedWindowAttention(Module):
    """Shifted Window Attention.

    (Shifted) Window Attention calculates attention over windows of the input.
    It was introduced in Swin Transformer [SWIN]_ and is used in Uformer.

    References
    ----------
    .. [SWIN] Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." ICCV 2021.
    """

    rel_position_index: torch.Tensor

    def __init__(self, dim: int, channels: int, n_heads: int, window_size: int = 7, shifted: bool = True):
        """Initialize the ShiftedWindowAttention module.

        Parameters
        ----------
        dim : int
            The dimension of the input.
        channels : int
            The number of channels in the input.
        n_heads : int
            The number of attention heads. The number if channels per head is ``channels // n_heads``.
        window_size : int
            The size of the window.
        shifted : bool
            Whether to shift the window.
        """
        super().__init__()
        if channels % n_heads:
            raise ValueError('channels must be divisible by n_heads.')
        self.channels = channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.shifted = shifted
        self.to_qkv = ConvND(dim)(channels, 3 * channels, 1)
        self.dim = dim
        coords_1d = torch.arange(window_size)
        coords_nd = torch.stack(torch.meshgrid(*([coords_1d] * dim), indexing='ij'), 0).flatten(1)
        rel_coords = coords_nd[:, :, None] - coords_nd[:, None, :]  # (dim, window_size**dim, window_size**dim)
        rel_coords += window_size - 1  # shift to >=0
        rel_position_index = ravel_multi_index(tuple(rel_coords), (2 * window_size - 1,) * dim)
        self.register_buffer('rel_position_index', rel_position_index)

        self.relative_position_bias_table = torch.nn.Parameter(torch.empty((2 * window_size - 1) ** dim, n_heads))
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02, a=-0.04, b=0.04)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ShiftedWindowAttention.

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
        """Apply the ShiftedWindowAttention."""
        if self.shifted:
            x = torch.roll(x, (-(self.window_size // 2),) * self.dim, dims=tuple(range(-self.dim, 0)))
        qkv = self.to_qkv(x)
        windowed = sliding_window(qkv, window_shape=self.window_size, stride=self.window_size, dim=range(-self.dim, 0))
        flat = windowed.flatten(0, self.dim - 1).flatten(-self.dim)
        q, k, v = rearrange(
            flat,
            'spatial batch (qkv heads channels) window->qkv spatial batch heads window channels',
            heads=self.n_heads,
            qkv=3,
        )
        bias = rearrange(self.relative_position_bias_table[self.rel_position_index], 'wd1 wd2 heads -> 1 heads wd1 wd2')
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        result = rearrange(result, 'spatial batch head window channels->batch (head channels) spatial window')
        result = result.unflatten(-2, windowed.shape[: self.dim]).unflatten(-1, (self.window_size,) * self.dim)
        # permute (in 3d) batch channels z y x wz wy wx -> batch channels wz z wy y wx x
        result = result.moveaxis(list(range(-self.dim, 0)), list(range(3, 3 + 2 * self.dim, 2)))
        result = result.reshape(x.shape)
        if self.shifted:
            result = torch.roll(result, (self.window_size // 2,) * self.dim, dims=tuple(range(-self.dim, 0)))
        return result


''
