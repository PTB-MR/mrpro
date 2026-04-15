"""Class for Convolutional Dictionary Transforms (Analysis and Synthesis)."""

from typing import Literal

import torch
from einops import pack, rearrange, unpack

from mrpro.operators.LinearOperator import LinearOperator


def prepare_conv_op_input(
    input_tensor: torch.Tensor,
    kernel: torch.Tensor,
    mode: Literal['analysis', 'synthesis'],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare input tensor and kernel for the application of the convolutional dictionary operator.

    Parameters
    ----------
    input_tensor
        Input tensor to be prepared.
    kernel
        Convolutional kernel.
    mode
        Mode to use for the convolutional operation.

    Returns
    -------
        Prepared input tensor, kernel and the number of groups for the convolutional operation.
    """
    if input_tensor.is_complex() and not kernel.is_complex():
        groups = 2
        n_convolution_dimensions = kernel.ndim - 1
        input_tensor = torch.cat([input_tensor.real, input_tensor.imag], dim=-(n_convolution_dimensions + 1))
        if mode == 'analysis':
            kernel = torch.cat(2 * [kernel], dim=0).to(input_tensor.device).unsqueeze(1)
        else:
            kernel = torch.stack(2 * [kernel], dim=0).to(input_tensor.device)
    else:
        groups = 1
        unsqueeze_dim = 0 if mode == 'synthesis' else 1
        kernel = kernel.unsqueeze(unsqueeze_dim)

    return input_tensor, kernel, groups


def reshape_input(
    input_tensor: torch.Tensor,
    operation: Literal['forward', 'adjoint'],
    mode: Literal['analysis', 'synthesis'],
    n_convolution_dimensions: int,
) -> tuple[torch.Tensor, list[tuple[int, ...] | list[int]]]:
    """Reshape input tensor for application of the convolutional dictionary operator.

    Parameters
    ----------
    input_tensor
        Input tensor to be reshaped.
    operation
        Operation of the convolutional operation ("forward" or "adjoint").
    mode
        Mode of the convolutional operation ("analysis" or "synthesis").
    n_convolution_dimensions
        Number of dimensions of the convolutional kernel (e.g. 2 for 2D kernels).

    Returns
    -------
        reshaped input tensor.
    """
    spatial_dimensions = ' '.join(f's{i}' for i in range(n_convolution_dimensions))
    if operation == 'forward':
        if mode == 'synthesis':
            input_tensor = rearrange(input_tensor, f'maps ... {spatial_dimensions} -> ... maps {spatial_dimensions}')
        else:
            input_tensor = input_tensor.unsqueeze(-(n_convolution_dimensions + 1))
    else:  # adjoint
        if mode == 'analysis':
            input_tensor = rearrange(input_tensor, f'maps ... {spatial_dimensions} -> ... maps {spatial_dimensions}')
        else:
            input_tensor = input_tensor.unsqueeze(-(n_convolution_dimensions + 1))

    input_tensor, ps = pack([input_tensor], f' * maps {spatial_dimensions}')

    return input_tensor, ps


def undo_reshape_input(
    input_tensor: torch.Tensor,
    ps: list[tuple[int, ...] | list[int]],
    operation: Literal['forward', 'adjoint'],
    mode: Literal['analysis', 'synthesis'],
    n_convolution_dimensions: int,
    groups: int,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Undo the reshaping of the input tensor for application of the convolutional dictionary operator.

    Parameters
    ----------
    input_tensor
        Input tensor to be reshaped.
    ps
        List of shapes for unpacking the tensor.
    operation
        Operation of the convolutional operation ("forward" or "adjoint").
    mode
        Mode of the convolutional operation ("analysis" or "synthesis").
    n_convolution_dimensions
        Number of dimensions of the convolutional kernel (e.g. 2 for 2D kernels).
    groups
        Number of groups that were used for the convolutional operation.
    kernel
        Convolutional kernel.

    Returns
    -------
        reshaped input tensor.
    """
    spatial_dimensions = ' '.join(f's{i}' for i in range(n_convolution_dimensions))
    [input_tensor] = unpack(input_tensor, ps, f'* maps {spatial_dimensions}')
    if operation == 'forward':
        if mode == 'synthesis':
            if groups == 1:
                input_tensor = input_tensor.squeeze(-(n_convolution_dimensions + 1))
            else:
                input_tensor = torch.view_as_complex(
                    rearrange(
                        input_tensor,
                        f'... channels {spatial_dimensions} -> ... {spatial_dimensions} channels',
                    ).contiguous()
                )
        else:
            input_tensor = rearrange(input_tensor, f'... maps {spatial_dimensions} -> maps ... {spatial_dimensions}')
            if groups == 2:
                input_tensor = input_tensor[: kernel.shape[0], ...] + 1j * input_tensor[kernel.shape[0] :, ...]

    else:  # adjoint
        if mode == 'analysis':
            if groups == 1:
                input_tensor = input_tensor.squeeze(-(n_convolution_dimensions + 1))
            else:
                input_tensor = torch.view_as_complex(
                    rearrange(
                        input_tensor,
                        f'... channels {spatial_dimensions} -> ... {spatial_dimensions} channels',
                    ).contiguous()
                )
        else:
            input_tensor = rearrange(input_tensor, f'... maps {spatial_dimensions} -> maps ... {spatial_dimensions}')
            if groups == 2:
                input_tensor = input_tensor[: kernel.shape[0], ...] + 1j * input_tensor[kernel.shape[0] :, ...]

    return input_tensor


class ConvDictionaryOp(LinearOperator):
    """Convolutional Dictionary operator class."""

    def __init__(
        self,
        kernel: torch.Tensor,
        mode: Literal['analysis', 'synthesis'],
        pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'circular',
    ) -> None:
        """Convolutional Dictionary Operator class.

        The operator implements the application of a convolutional transform (either in analysis or in synthesis mode)
        to an input tensor.
        Thereby, the following cases are supported: if the filter is real-valued, the transform can
        be applied to both real and complex-valued inputs. For complex-valued inputs, the same filter is applied to
        the real and the imaginary part. If the filter is complex-valued, the transform can only be applied to
        complex-valued inputs.

        Parameters
        ----------
        kernel
            Convolutional filter of shape (n_filters, *spatial_dims). The filter filter dimension is specified
            by the number of dimension in the *spatial_dims. Currently, only 1D, 2D and 3D filters are supported.
            Example: for 2D filters, the shape is (n_filters, ky, kx), for 3D filters, (n_filters, kz, ky, kx).
            Also, all spatial dimensions of the kernels should be odd.
        mode
            the mode of the convolutional dictionary operator, either "analysis" or "synthesis"
        pad_mode
            the mode to use for padding
        """
        super().__init__()
        self.n_convolution_dimensions = kernel.ndim - 1
        if self.n_convolution_dimensions not in (1, 2, 3):
            raise ValueError(
                f'Only 1D, 2D and 3D filters are supported, but kernel \
                    has {self.n_convolution_dimensions} spatial dimensions.'
            )
        if any(k % 2 == 0 for k in kernel.shape[-self.n_convolution_dimensions :]):
            raise ValueError(
                f'All spatial dimensions of the kernel must be odd, but got kernel with shape {kernel.shape}.'
            )
        if mode not in ('analysis', 'synthesis'):
            raise ValueError(f"Mode must be either 'analysis' or 'synthesis', but got {mode}.")
        if pad_mode not in ('constant', 'reflect', 'replicate', 'circular'):
            raise ValueError(
                f"Pad mode must be either 'constant', 'reflect', 'replicate', or 'circular', but got {pad_mode}."
            )
        if self.n_convolution_dimensions == 1:
            self.conv_op = torch.nn.functional.conv1d
            self.conv_op_adjoint = torch.nn.functional.conv_transpose1d
        elif self.n_convolution_dimensions == 2:
            self.conv_op = torch.nn.functional.conv2d
            self.conv_op_adjoint = torch.nn.functional.conv_transpose2d
        elif self.n_convolution_dimensions == 3:
            self.conv_op = torch.nn.functional.conv3d
            self.conv_op_adjoint = torch.nn.functional.conv_transpose3d

        self.kernel = torch.nn.Parameter(kernel, requires_grad=kernel.requires_grad)
        self.mode = mode
        self.pad_mode = pad_mode

        self.padding = tuple(k // 2 for k in kernel.shape[-self.n_convolution_dimensions :])

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the convolutional dictionary transform.

        This functions first appropriately reshapes the input tensor, then appropriately pads it and performs the
        convolution. Finally, it undoes the reshaping to maintain the original spatial dimensions.

        Parameters
        ----------
        x
            Input tensor.
        filter
            Convolutional filter of shape (n_filters, *spatial_dims). The filter filter dimension is specified
            by the number of dimension in the *spatial_dims. Currently, only 1D, 2D and 3D filters are supported.

        Returns
        -------
            The result of the convolutional dictionary applied to the input.


        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of ConvDictionary.

        .. note::
            Prefer calling the instance of the ConvDictionary operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if self.mode == 'synthesis' and x.shape[0] != self.kernel.shape[0]:
            raise ValueError(
                'First dimension of input must match the number of filters in the kernel \
                for the forward in synthesis mode.'
            )
        if self.kernel.is_complex() and not x.is_complex():
            raise ValueError('Input tensor must be complex-valued when the kernel is complex-valued.')

        x, ps = reshape_input(
            x,
            operation='forward',
            mode=self.mode,
            n_convolution_dimensions=self.n_convolution_dimensions,
        )

        x, kernel, groups = prepare_conv_op_input(x, self.kernel, self.mode)

        x = self.conv_op(x, kernel, groups=groups, padding=self.padding)

        x = undo_reshape_input(
            x,
            ps,
            operation='forward',
            mode=self.mode,
            n_convolution_dimensions=self.n_convolution_dimensions,
            groups=groups,
            kernel=self.kernel,
        )

        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the convolutional dictionary transform.

        This operation is the adjoint of the forward `ConvDictionary`. If the forward
        operation was a synthesis transform, the adjoint is an analysis transform. If the forward was
        an analysis transform, the adjoint is a synthesis transform.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The transformed input tensor.
        """
        if self.mode == 'analysis' and x.shape[0] != self.kernel.shape[0]:
            raise ValueError(
                'First dimension of input must match the number of filters in the kernel \
                for the adjoint in analysis mode.'
            )
        if self.kernel.is_complex() and not x.is_complex():
            raise ValueError('Input tensor must be complex-valued when the kernel is complex-valued.')

        x, ps = reshape_input(
            x,
            operation='adjoint',
            mode=self.mode,
            n_convolution_dimensions=self.n_convolution_dimensions,
        )

        x, kernel, groups = prepare_conv_op_input(x, self.kernel, self.mode)

        x = self.conv_op_adjoint(x, kernel.conj(), groups=groups, padding=self.padding)

        x = undo_reshape_input(
            x,
            ps,
            operation='adjoint',
            mode=self.mode,
            n_convolution_dimensions=self.n_convolution_dimensions,
            groups=groups,
            kernel=self.kernel,
        )

        return (x,)
