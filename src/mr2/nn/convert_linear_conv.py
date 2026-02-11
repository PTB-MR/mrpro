"""Convert Linear layers to kernel size 1 ConvNd layers and vice versa."""

from typing import Literal, overload

import torch
from torch.nn import Conv1d, Conv2d, Conv3d, Linear

from mr2.nn.ndmodules import convND


@overload
def linear_to_conv(linear_layer: Linear, n_dim: Literal[1]) -> Conv1d: ...


@overload
def linear_to_conv(linear_layer: Linear, n_dim: Literal[2]) -> Conv2d: ...


@overload
def linear_to_conv(linear_layer: Linear, n_dim: Literal[3]) -> Conv3d: ...


@overload
def linear_to_conv(linear_layer: Linear, n_dim: int) -> Conv1d | Conv2d | Conv3d: ...


def linear_to_conv(linear_layer: Linear, n_dim: int) -> Conv1d | Conv2d | Conv3d:
    """Convert a Linear layer to a ConvNd layer with kernel size 1.

    Rearranging the spatial dimensions to the batch dimension,
    applying the linear layer and rearranging the spatial dimensions back
    is equivalent to applying a kernel size 1 ConvNd layer.

    This function will create the Conv1d, Conv2d, or Conv3d with the correct weights and bias.

    See :func:`conv_to_linear` for the reverse operation.



    Parameters
    ----------
    linear_layer
        The linear layer to convert.
    n_dim
        The convolution dimension (1, 2, or 3).

    Returns
    -------
        A Conv layer with equivalent weights and bias.
    """
    conv = convND(n_dim)(
        in_channels=linear_layer.in_features,
        out_channels=linear_layer.out_features,
        kernel_size=1,
        bias=linear_layer.bias is not None,
        device=linear_layer.weight.device,
        dtype=linear_layer.weight.dtype,
    )

    with torch.no_grad():
        conv.weight.copy_(linear_layer.weight.view_as(conv.weight))
        if conv.bias is not None and linear_layer.bias is not None:
            conv.bias.copy_(linear_layer.bias)

    return conv


def conv_to_linear(conv_layer: Conv1d | Conv2d | Conv3d) -> Linear:
    """
    Convert a Conv1d, Conv2d, or Conv3d layer with kernel size 1 to a Linear layer.

    Applying a kernel size 1 ConvNd layer is equivalent to applying a Linear layer to each voxel.
    This function will create the Linear layer with the correct weights and bias.

    See :func:`linear_to_conv` for the reverse operation.

    Parameters
    ----------
    conv_layer : nn.Module
        The convolutional layer to convert. Must have kernel size 1.

    Returns
    -------
        A linear layer with equivalent weights and bias.
    """
    if not all(k == 1 for k in conv_layer.kernel_size):
        raise ValueError('Kernel size must be 1 for conversion.')
    linear = Linear(
        conv_layer.in_channels,
        conv_layer.out_channels,
        bias=conv_layer.bias is not None,
        device=conv_layer.weight.device,
        dtype=conv_layer.weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(conv_layer.weight.view_as(linear.weight))
        if linear.bias is not None and conv_layer.bias is not None:
            linear.bias.copy_(conv_layer.bias)

    return linear
