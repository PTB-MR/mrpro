"""Neighborhood Self Attention."""

from collections.abc import Sequence
from functools import cache, reduce
from typing import TYPE_CHECKING, TypeVar, cast

import torch
from einops import rearrange
from packaging.version import parse as parse_version
from torch.nn import Linear, Module

from mr2.nn.AxialRoPE import AxialRoPE
from mr2.utils.to_tuple import to_tuple

T = TypeVar('T')

if TYPE_CHECKING or parse_version(torch.__version__) >= parse_version('2.6'):
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
else:

    class BlockMask:
        """Dummy class for older PyTorch versions."""


_compiled_flex_attention = torch.compile(
    lambda q, k, v, mask: flex_attention(q, k, v, block_mask=mask),
    dynamic=False,
)


@torch.compiler.disable
@cache
def neighborhood_mask(
    device: str,
    input_size: torch.Size,
    kernel_size: int | tuple[int, ...],
    dilation: int | tuple[int, ...] = 1,
    circular: bool | tuple[bool, ...] = False,
) -> BlockMask:  # pragma: no cover
    """Create a flex attention block mask for neighborhood attention.

    This function defines which key/value pairs a query can attend to based
    on a local neighborhood. The neighborhood is defined by `kernel_size`
    and `dilation` and can be circular (wrapping around edges).

    Parameters
    ----------
    input_size
        The dimensions of the input tensor (e.g., (H, W) for 2D).
    kernel_size
        The size of the attention neighborhood window. Can be a single
        integer for a symmetric window or a sequence of integers for
        each dimension.
    dilation
        The dilation factor for the neighborhood
        Can be a single integer for a symmetric window or a sequence
        of integers for each dimension.
    circular
        Whether the neighborhood wraps around the edges (circular padding).
        Can be a single boolean or a sequence of booleans.
    device
        The device to create the mask on.

    Returns
    -------
        A mask object suitable for `flex_attention` that defines the
        allowed attention connections.
    """
    kernel_size_tuple, dilation_tuple, circular_tuple = (
        to_tuple(len(input_size), x) for x in (kernel_size, dilation, circular)
    )

    def unravel_index(idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Convert a flat 1D index into multi-dimensional coordinates."""
        idx = idx.clone()
        coords = []
        for dim in reversed(input_size):
            coords.append(idx % dim)
            idx = torch.div(idx, dim, rounding_mode='floor').long()
        coords.reverse()
        return tuple(coords)

    def mask(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Determine if a query can attend to a key/value pair."""
        q_coord = unravel_index(q_idx)
        kv_coord = unravel_index(kv_idx)

        masks = []
        for input_, kernel_, dilation_, circular_, q_, kv_ in zip(
            input_size,
            kernel_size_tuple,
            dilation_tuple,
            circular_tuple,
            q_coord,
            kv_coord,
            strict=False,
        ):
            masks.append((q_ % dilation_) == (kv_ % dilation_))
            kernel_dilation = kernel_ * dilation_
            window_left = kernel_dilation // 2
            window_right = (kernel_dilation // 2) + ((kernel_dilation % 2) - 1)
            if circular_:
                left = (q_ - kv_ + input_) % input_
                right = (kv_ - q_ + input_) % input_
                masks.append((left <= window_left) | (right <= window_right))
            else:
                center = q_.clamp(window_left, input_ - 1 - window_right)
                left = center - kv_
                right = kv_ - center
                masks.append(((left >= 0) & (left <= window_left)) | ((right >= 0) & (right <= window_right)))
        return reduce(lambda x, y: x & y, masks)

    qkv_len = input_size.numel()
    return create_block_mask(mask, B=None, H=None, Q_LEN=qkv_len, KV_LEN=qkv_len, device=torch.device(device))


class NeighborhoodSelfAttention(Module):
    """Attention where each query attends to a neighborhood of the key and value.

    Neighborhood attention is a type of attention where each query attends to a neighborhood of the key and value.
    It is a more efficient alternative to regular attention, especially for large input sizes [NAT]_.

    This implementation uses `~torch.nn.attention.flex_attention`. For a more efficient implementation,
    see also [NATTEN]_.


    References
    ----------
    .. [NAT] Hassani, A. et al. "Neighborhood Attention Transformer" CVPR, 2023, https://arxiv.org/abs/2204.07143
    .. [NATTEN] https://github.com/SHI-Labs/NATTEN/
    """

    n_head: int
    kernel_size: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    circular: bool | tuple[bool, ...]
    features_last: bool

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_heads: int,
        kernel_size: int | Sequence[int],
        dilation: int | Sequence[int] = 1,
        circular: bool | Sequence[bool] = False,
        features_last: bool = False,
        rope_embed_fraction: float = 1.0,
    ) -> None:
        """Initialize a neighborhood attention module.

        The parameters `kernel_size`, `dilation`, and `circular` can either be sequences, interpreted as per-dimension
        values, or scalars, interpreted as the same value for all dimensions.

        Parameters
        ----------
        n_channels_in
            The number of channels in the input tensor.
        n_channels_out
            The number of channels in the output tensor.
        n_heads
            The number of attention heads.
        kernel_size
            The size of the attention neighborhood window.
        dilation
            The dilation factor for the neighborhood.
        circular
            Whether the neighborhood wraps around the edges (circular padding)
        features_last
            Whether the channels are in the last dimension of the tensor, as common in visíon transformers.
            Otherwise, assume the channels are in the second dimension, as common in CNN models.
        rope_embed_fraction
            Fraction of channels to embed with RoPE.

        """
        if parse_version(torch.__version__) < parse_version('2.6.0'):
            raise NotImplementedError('NeighborhoodSelfAttention requires PyTorch 2.6.0 or higher')
        super().__init__()
        self.n_head = n_heads
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else tuple(kernel_size)
        self.dilation = dilation if isinstance(dilation, int) else tuple(dilation)
        self.circular = circular if isinstance(circular, bool) else tuple(circular)
        self.features_last = features_last
        channels_per_head = n_channels_in // n_heads
        self.to_qkv = Linear(n_channels_in, 3 * channels_per_head * n_heads)
        self.to_out = Linear(channels_per_head * n_heads, n_channels_out)
        self.rope = AxialRoPE(rope_embed_fraction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply neighborhood attention to the input tensor.

        Parameters
        ----------
        x
            The input tensor, with shape `(batch, channels, *spatial_dims)`
            or `(batch, *spatial_dims, channels)` (if `features_last`).

        Returns
        -------
            The output tensor after attention, with the same shape as the input tensor.
        """
        if not self.features_last:
            x = x.moveaxis(1, -1)
        spatial_shape = x.shape[1:-1]
        qkv = self.to_qkv(x)
        query, key, value = rearrange(
            qkv,
            'batch ... (qkv heads channels) -> qkv batch heads (...) channels',
            qkv=3,
            heads=self.n_head,
        )
        query, key = self.rope(query, key)
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        device = str(qkv.device)
        mask = neighborhood_mask(
            device=device,
            input_size=spatial_shape,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            circular=self.circular,
        )
        mask = torch.compiler.assume_constant_result(mask)
        if torch.compiler.is_compiling():
            out = cast(torch.Tensor, flex_attention(query, key, value, block_mask=mask))
        else:
            out = cast(torch.Tensor, _compiled_flex_attention(query, key, value, mask))
        out = rearrange(out, 'batch head sequence channels -> batch sequence (head channels)')
        out = self.to_out(out)
        out = out.unflatten(-2, spatial_shape)
        if not self.features_last:
            out = out.moveaxis(-1, 1)
        return out
