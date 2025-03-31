from collections.abc import Sequence

import torch
import torch.nn.functional as F

from mrpro.operators import LinearOperator


def _normalize_param(param: int | Sequence[int], n: int, name: str) -> tuple[int, ...]:
    if isinstance(param, int):
        return (param,) * n
    elif len(param) != n:
        raise ValueError(f"'{name}' must be an int or a sequence of length {n}")
    param_ = tuple(param)
    if any(p <= 0 for p in param_):
        raise ValueError(f'{name} must be >= 0.')
    return param_


def conv_nd(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Nd convolution."""
    ndim = x.dim() - 2
    stride = _normalize_param(stride, ndim, 'stride')
    padding = _normalize_param(padding, ndim, 'padding')
    dilation = _normalize_param(dilation, ndim, 'dilation')
    if ndim <= 3:
        if ndim == 1:
            return F.conv1d(x, weight, None, stride, padding, dilation, groups)
        if ndim == 2:
            return F.conv2d(x, weight, None, stride, padding, dilation, groups)
        return F.conv3d(x, weight, None, stride, padding, dilation, groups)
    # TODO: allow multiple batch dims
    n_batch, channels_in = x.shape[:2]
    spatial = x.shape[2:]
    channels_out, *_ = weight.shape
    ks = weight.shape[2:]
    out_dims = [((spatial[i] + 2 * padding[i] - dilation[i] * (ks[i] - 1) - 1) // stride[i] + 1) for i in range(ndim)]
    on, kn = out_dims[-1], ks[-1]
    pad_tuple = [p for p in reversed(padding) for _ in (0, 1)]
    padded = F.pad(x, pad_tuple)
    base_idx = torch.arange(on, device=x.device) * stride[-1]
    offsets = torch.arange(kn, device=x.device) * dilation[-1]
    indices = base_idx.unsqueeze(1) + offsets.unsqueeze(0)
    # padded has shape [B, Cin, *padded_spatial] where padded_spatial has length ndim.
    # Advanced indexing on the last dim gives shape [B, Cin, *padded_spatial[:-1], on, kn]
    slices = padded[..., indices]
    slices = slices.permute(0, ndim + 1, ndim + 2, 1, *range(2, ndim + 1))
    slices = slices.reshape(n_batch * on * kn, channels_in, *padded.shape[2:-1])
    weight_reshaped = weight.permute(0, ndim + 1, 1, *range(2, ndim + 1))
    weight_reshaped = weight_reshaped.reshape(channels_out * kn, channels_in // groups, *ks[:-1])
    conv_res = conv_nd(slices, weight_reshaped, stride[:-1], 0, dilation[:-1], groups * kn)
    conv_res = conv_res.view(n_batch, on, kn, channels_out, *out_dims[:-1]).sum(dim=2)
    conv_res = conv_res.permute(0, 2, *range(3, conv_res.dim()), 1)
    return conv_res


def conv_transposed_nd(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    output_padding: int | Sequence[int] = 0,
    groups: int = 1,
    dilation: int | Sequence[int] = 1,
) -> torch.Tensor:
    """Nd transposed convolution"""
    ndim = x.dim() - 2
    stride = _normalize_param(stride, ndim, 'stride')
    padding = _normalize_param(padding, ndim, 'padding')
    output_padding = _normalize_param(output_padding, ndim, 'output_padding')
    dilation = _normalize_param(dilation, ndim, 'dilation')
    if ndim <= 3:
        if ndim == 1:
            return F.conv_transpose1d(x, weight, None, stride, padding, output_padding, groups, dilation)
        if ndim == 2:
            return F.conv_transpose2d(x, weight, None, stride, padding, output_padding, groups, dilation)
        return F.conv_transpose3d(x, weight, None, stride, padding, output_padding, groups, dilation)
    # TODO: multiple batch dims
    n_batch, channels_in = x.shape[:2]
    spatial = x.shape[2:]
    ks = weight.shape[2:]
    channels_out = weight.shape[1] * groups
    out_dims = [
        ((spatial[i] - 1) * stride[i] - 2 * padding[i] + dilation[i] * (ks[i] - 1) + output_padding[i] + 1)
        for i in range(ndim)
    ]
    in_last, kn, out_last = spatial[-1], ks[-1], out_dims[-1]
    reshaped = x.permute(0, x.dim() - 1, 1, *range(2, x.dim() - 1))
    reshaped = reshaped.reshape(n_batch * in_last, channels_in, *spatial[:-1])
    weight_r = weight.permute(0, ndim + 1, 1, *range(2, ndim + 1))
    weight_r = weight_r.reshape(channels_in * kn, channels_out // groups, *ks[:-1])
    trans = conv_transposed_nd(
        reshaped, weight_r, stride[:-1], padding[:-1], output_padding[:-1], groups * kn, dilation[:-1]
    )
    trans = trans.view(n_batch, in_last, kn, channels_out, *out_dims[:-1])
    out = torch.zeros(n_batch, channels_out, *out_dims, device=x.device, dtype=x.dtype)
    for i in range(in_last):
        start = i * stride[-1] - padding[-1]
        for j in range(kn):
            pos = start + j * dilation[-1]
            if 0 <= pos < out_last:
                out[..., pos] += trans[:, i, j]
    return out


class ConvolutionOp(LinearOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        ndim: int,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
    ) -> None:
        """Initialize convolution operator."""
        super().__init__()
        self.ndim = ndim
        self.groups = groups
        self.stride = _normalize_param(stride, ndim, 'stride')
        self.padding = _normalize_param(padding, ndim, 'padding')
        self.dilation = _normalize_param(dilation, ndim, 'dilation')
        self.weight = weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the convolution."""
        return (conv_nd(x, self.weight, self.stride, self.padding, self.dilation, self.groups),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the adjoint convolution."""
        return (conv_transposed_nd(x, self.weight, None, self.stride, self.padding, 0, self.groups, self.dilation),)
