from abc import ABC
from collections.abc import Sequence
from functools import partial

import torch
from einops import rearrange
from torch.nn import Identity, Linear, Module, Parameter, ReLU, Sequential, Sigmoid, SiLU

from mrpro.utils.reshape import unsqueeze_tensors_right


class NDModule(Module, ABC):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module to the input tensor."""
        return super().__call__(x)

    def __forward__(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class ConvND(NDModule):
    """N-dimensional convolution.

    Parameters
    ----------
    dim
        The dimension of the convolution.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int = 1,
        padding: str | Sequence[int] | int = 'same',
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
    ) -> None:
        if not isinstance(kernel_size, int) and len(kernel_size) != dim:
            raise ValueError(f'kernel_size must be an int or a sequence of length {dim}')
        if stride is not None and not isinstance(stride, int) and len(stride) != dim:
            raise ValueError(f'stride must be None, an int, or a sequence of length {dim}')
        if padding != 'same' and not isinstance(padding, int) and len(padding) != dim:
            raise ValueError(f'padding must be an int or a sequence of length {dim}')
        try:
            self.module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[dim](
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
            )
        except KeyError:
            raise NotImplementedError(f'ConvND for dim {dim} not implemented.') from None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner(x)


class MaxPoolND(NDModule):
    def __init__(self, dim: int) -> None:
        super().__init__()
        try:
            self.module = {1: torch.nn.MaxPool1d, 2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}[dim]
        except KeyError:
            raise NotImplementedError(f'MaxPoolNd for dim {dim} not implemented.')


class AvgPoolND(NDModule):
    """N-dimensional average pooling."""

    def __init__(
        self,
        dim: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = False,
        divisor_override: int | None = None,
    ) -> None:
        """Parameters for AvgPoolNd.

        Parameters
        ----------
        dim
            The dimension of the input tensor.
        kernel_size
            The size of the kernel.
        stride
            The stride of the kernel.
        padding
            The padding of the kernel.
        ceil_mode
            Whether to use ceil instead of floor to compute the output shape.
        count_include_pad
            Whether to include the padding in the divisor.
        divisor_override
            Overwrite the default divisor of the number of elements in the pooling region.
        """
        super().__init__()
        if not isinstance(kernel_size, int) and len(kernel_size) != dim:
            raise ValueError(f'kernel_size must be an int or a sequence of length {dim}')
        if stride is not None and not isinstance(stride, int) and len(stride) != dim:
            raise ValueError(f'stride must be None, an int, or a sequence of length {dim}')
        if padding != 'same' and not isinstance(padding, int) and len(padding) != dim:
            raise ValueError(f'padding must be an int or a sequence of length {dim}')
        try:
            module = {1: torch.nn.AvgPool1d, 2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d()}[dim]
        except KeyError:
            raise NotImplementedError(f'AvgPoolNd for dim {dim} not implemented.') from None
        self.module = module(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


class AdaptiveAvgPoolND(NDModule):
    """N-dimensional adaptive average pooling."""

    def __init__(self, dim: int, output_size: int | None | Sequence[int] = None):
        super().__init__()
        if not isinstance(output_size, int) and len(output_size) != dim:
            raise ValueError(f'output_size must be an int or a sequence of length {dim}')
        try:
            self.module = (torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d)[dim - 1](
                output_size
            )
        except KeyError:
            raise NotImplementedError(f'AdaptiveAvgPoolnD for dim {dim} not implemented.') from None


class MaxPoolND(NDModule):
    """N-dimensional max pooling."""

    def __init__(
        self,
        dim: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        ceil_mode: bool = False,
    ) -> None:
        """Initialize MaxPoolNd.

        Parameters
        ----------
        dim
            The dimension of the input tensor.
        kernel_size
            The size of the kernel.
        stride
            The stride of the kernel.
        padding
            The padding of the kernel.
        dilation
            The dilation of the kernel.
        ceil_mode
            Whether to use ceil instead of floor to compute the output shape.
        """
        if not isinstance(kernel_size, int) and len(kernel_size) != dim:
            raise ValueError(f'kernel_size must be an int or a sequence of length {dim}')
        if stride is not None and not isinstance(stride, int) and len(stride) != dim:
            raise ValueError(f'stride must be None, an int, or a sequence of length {dim}')
        if not isinstance(padding, int) and len(padding) != dim:
            raise ValueError(f'padding must be an int or a sequence of length {dim}')
        if not isinstance(dilation, int) and len(dilation) != dim:
            raise ValueError(f'dilation must be an int or a sequence of length {dim}')
        super().__init__()
        self.module = {1: torch.nn.MaxPool1d, 2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}[dim](
            kernel_size, stride, padding, dilation, ceil_mode
        )
