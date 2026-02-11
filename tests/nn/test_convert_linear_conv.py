"""Tests for converting between Linear and Conv layers."""

from typing import Literal

import pytest
import torch
from mr2.nn.convert_linear_conv import conv_to_linear, linear_to_conv
from mr2.utils import RandomGenerator
from torch.nn import Conv1d, Conv2d, Conv3d, Linear

DEVICES = pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
SHAPES = pytest.mark.parametrize(
    ('dim', 'channels_in', 'channels_out', 'bias'),
    [
        (1, 32, 64, True),
        (2, 16, 32, True),
        (3, 8, 16, True),
        (3, 1, 1, False),
    ],
    ids=['1d', '2d', '3d', '3d_no_bias'],
)


@SHAPES
@DEVICES
def test_linear_to_conv(device: str, dim: Literal[1, 2, 3], channels_in: int, channels_out: int, bias: bool) -> None:
    """Test converting Linear to Conv layer."""
    rng = RandomGenerator(seed=42)
    linear = Linear(channels_in, channels_out, bias=bias).to(device)
    linear.weight.data = rng.rand_like(linear.weight)
    if bias:
        linear.bias.data = rng.rand_like(linear.bias)

    conv = linear_to_conv(linear, dim)
    assert isinstance(conv, (Conv1d, Conv2d, Conv3d)[dim - 1])

    assert conv.in_channels == channels_in
    assert conv.out_channels == channels_out
    assert conv.kernel_size == (1,) * dim
    assert conv.bias is not None if bias else conv.bias is None

    assert conv.weight.device.type == device
    if conv.bias is not None:
        assert conv.bias.device.type == device


@SHAPES
def test_linear_to_conv_functional(dim: Literal[1, 2, 3], channels_in: int, channels_out: int, bias: bool) -> None:
    """Test functional equivalence of Linear to Conv conversion."""
    rng = RandomGenerator(seed=42)
    linear = Linear(channels_in, channels_out, bias=bias)
    linear.weight.data = rng.rand_like(linear.weight)
    if bias:
        linear.bias.data = rng.rand_like(linear.bias)

    conv = linear_to_conv(linear, dim)
    spatial_shape = (4,) * dim
    x = rng.randn_tensor((2, channels_in, *spatial_shape), torch.float32)

    y_conv = conv(x)
    y_conv = y_conv.moveaxis(1, -1).flatten(0, -2)

    x_reshaped = x.moveaxis(1, -1).flatten(0, -2)
    y_linear = linear(x_reshaped)

    torch.testing.assert_close(y_conv, y_linear)


@SHAPES
@DEVICES
def test_conv_to_linear(device: str, dim: Literal[1, 2, 3], channels_in: int, channels_out: int, bias: bool) -> None:
    """Test converting Conv layer to Linear."""
    rng = RandomGenerator(seed=42)
    conv_class = (Conv1d, Conv2d, Conv3d)[dim - 1]
    conv = conv_class(channels_in, channels_out, kernel_size=1, bias=bias).to(device)
    conv.weight.data = rng.rand_like(conv.weight)
    if conv.bias is not None:
        conv.bias.data = rng.rand_like(conv.bias)

    linear = conv_to_linear(conv)

    assert isinstance(linear, Linear)
    assert linear.in_features == channels_in
    assert linear.out_features == channels_out
    assert linear.bias is not None if bias else linear.bias is None

    assert linear.weight.device.type == device
    if bias:
        assert linear.bias.device.type == device


@SHAPES
def test_conv_to_linear_functional(dim: Literal[1, 2, 3], channels_in: int, channels_out: int, bias: bool) -> None:
    """Test functional equivalence of Conv to Linear conversion."""
    rng = RandomGenerator(seed=42)
    conv_class = (Conv1d, Conv2d, Conv3d)[dim - 1]
    conv = conv_class(channels_in, channels_out, kernel_size=1, bias=bias)
    conv.weight.data = rng.rand_like(conv.weight)
    if conv.bias is not None:
        conv.bias.data = rng.rand_like(conv.bias)

    linear = conv_to_linear(conv)
    spatial_shape = (4,) * dim

    x = rng.randn_tensor((2, channels_in, *spatial_shape), torch.float32)
    y_conv = conv(x)
    y_conv = y_conv.moveaxis(1, -1).flatten(0, -2)

    x_reshaped = x.moveaxis(1, -1).flatten(0, -2)
    y_linear = linear(x_reshaped)

    torch.testing.assert_close(y_conv, y_linear)


def test_conv_to_linear_invalid_kernel() -> None:
    """Test conv_to_linear with invalid kernel size."""
    conv = Conv2d(32, 64, kernel_size=3, bias=True)
    with pytest.raises(ValueError, match='Kernel size must be 1'):
        conv_to_linear(conv)


@SHAPES
@DEVICES
def test_round_trip_conversion(
    device: str, dim: Literal[1, 2, 3], channels_in: int, channels_out: int, bias: bool
) -> None:
    """Test round-trip conversion between Linear and Conv layers."""
    rng = RandomGenerator(seed=42)

    linear1 = Linear(channels_in, channels_out, bias=bias).to(device)
    linear1.weight.data = rng.rand_like(linear1.weight)
    if bias:
        linear1.bias.data = rng.rand_like(linear1.bias)

    conv = linear_to_conv(linear1, dim)
    linear2 = conv_to_linear(conv)

    assert linear2.in_features == channels_in
    assert linear2.out_features == channels_out
    assert linear2.bias is not None if bias else linear2.bias is None

    torch.testing.assert_close(linear2.weight, linear1.weight)
    if bias:
        torch.testing.assert_close(linear2.bias, linear1.bias)
