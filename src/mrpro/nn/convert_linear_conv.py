"""Convert Linear layers to kernel size 1 ConvNd layers and vice versa."""

import torch
import torch.nn as nn
from torch.nn import Module, Conv1d, Conv2d, Conv3d, Linear
from mrpro.nn.ndmodules import ConvND
from typing import Literal, overload


@overload
def linear_to_conv(linear_layer: Linear, dim: Literal[1]) -> Conv1d: ...


@overload
def linear_to_conv(linear_layer: Linear, dim: Literal[2]) -> Conv2d: ...


@overload
def linear_to_conv(linear_layer: Linear, dim: Literal[3]) -> Conv3d: ...


@overload
def linear_to_conv(linear_layer: Linear, dim: int) -> Conv1d | Conv2d | Conv3d: ...


def linear_to_conv(linear_layer: Linear, dim: int) -> Conv1d | Conv2d | Conv3d:
    """Convert a Linear layer to a ConvNd layer with kernel size 1.

    Rearranging the spatial dimensions to the batch dimension, applying the linear layer and rearranging the spatial dimensions back
    it equivalent to applying the a kernel size 1 ConvNd layer.
    This function will create the ConvNd with the correct weights and bias.

    See :func:`conv_to_linear` for the reverse operation.



    Parameters
    ----------
    linear_layer : nn.Linear
        The linear layer to convert.
    dim : int
        The convolution dimension (1, 2, or 3).

    Returns
    -------
        A Conv layer with equivalent weights and bias.
    """
    conv = ConvND(dim)(
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
        conv_layer.weight.shape[0],
        conv_layer.weight.shape[1],
        bias=conv_layer.bias is not None,
        device=conv_layer.weight.device,
        dtype=conv_layer.weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(conv_layer.weight.view_as(linear.weight))
        if linear.bias is not None and conv_layer.bias is not None:
            linear.bias.copy_(conv_layer.bias)

    return linear
