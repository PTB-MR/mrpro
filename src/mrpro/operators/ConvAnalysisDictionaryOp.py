"""Convolutional Analysis Dictionary Operator."""

import warnings
from typing import Literal

import torch
from torchnd import adjoint_pad_nd, conv_nd, pad_nd

from mrpro.operators.LinearOperator import LinearOperator


class ConvAnalysisDictionaryOp(LinearOperator):
    """Convolutional Analysis Dictionary Operator."""

    def __init__(
        self,
        kernel: torch.Tensor,
        pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'circular',
    ) -> None:
        r"""Convolutional Analysis Dictionary Operator class.

        The operator implements the application of a convolutional analysis transform to an input tensor, i.e.
            :math:`Hx:=[h_1 \ast x, \ldots, h_K \ast x]^T`
        for convolutional filters :math:`h_1, \ldots, h_K` and input tensor :math:`x`.
        Thereby, for each :math:`k`, the operation is defined as
            :math:`h_k \ast x := \
                \mathrm{conv}(\mathrm{Re}(x), \mathrm{Re}(h_k)) \
                - \mathrm{conv}(\mathrm{Im}(x), \mathrm{Im}(h_k)) \
                + i \cdot (\mathrm{conv}(\mathrm{Re}(x), \mathrm{Im}(h_k)) \
                + \mathrm{conv}(\mathrm{Im}(x), \mathrm{Re}(h_k)))`,
        where :math:`\mathrm{conv}(\cdot, \cdot)` denotes the convolution operation.
        Thus, if the filter is real-valued and the input complex-valued, the same filter is applied to real
        and the imaginary part of the input.
        Note that, `\mathrm{conv}` actually performs a cross-correltation, matching torch's convolution implementation.

        Parameters
        ----------
        kernel
            Convolutional filter of shape (n_filters, *spatial_dims). The filter filter dimension is specified
            by the number of dimension in the *spatial_dims.
            Example: for 2D filters, the shape is (n_filters, ky, kx), for 3D filters, (n_filters, kz, ky, kx).
            Note that, typically, odd spatial dimensions are used for the kernels to avoid spatial pixel-shift artifacts
            in the result of the composition of the adjoint and the forward operator.
        pad_mode
            the mode to use for padding
        """
        super().__init__()
        if pad_mode not in ('constant', 'reflect', 'replicate', 'circular'):
            raise ValueError(
                f"Pad mode must be either 'constant', 'reflect', 'replicate', or 'circular', but got {pad_mode}."
            )
        self.kernel = kernel
        self.pad_mode = pad_mode

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the convolutional dictionary transform.

        This functions first appropriately reshapes the input tensor, then appropriately pads it and performs the
        convolution. Finally, it undoes the reshaping to maintain the original spatial dimensions.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The result of the convolutional analysis dictionary applied to the input.


        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of the convolutional analysis dictionary.

        .. note::
            Prefer calling the instance of the ConvAnalysisDictionaryOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if self.kernel.is_complex() and not x.is_complex():
            warnings.warn(
                'The input tensor is real-valued but the kernel is complex-valued. '
                'The input tensor will be treated as complex-valued with zero imaginary part.',
                UserWarning,
                stacklevel=2,
            )

        n_dim = self.kernel.ndim - 1
        pad = tuple(p for k in self.kernel.shape[1:] for p in (k // 2, k // 2))

        spatial_shape = x.shape[-n_dim:]
        batch_shape = x.shape[:-n_dim]

        x = x.reshape(-1, 1, *spatial_shape)
        x = pad_nd(x, pad=pad, dims=tuple(range(-n_dim, 0)), mode=self.pad_mode)
        y = conv_nd(x, self.kernel.unsqueeze(1), dim=tuple(range(-n_dim, 0)))
        y = y.movedim(1, 0).reshape(-1, *batch_shape, *spatial_shape)

        return (y,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the convolutional dictionary transform.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The transformed input tensor.
        """
        if x.shape[0] != self.kernel.shape[0]:
            raise ValueError('First dimension of input must match the number of filters in the kernel.')
        if self.kernel.is_complex() and not x.is_complex():
            warnings.warn(
                'The input tensor is real-valued but the kernel is complex-valued. '
                'The input tensor will be treated as complex-valued with zero imaginary part.',
                UserWarning,
                stacklevel=2,
            )

        n_dim = self.kernel.ndim - 1
        pad = tuple(p for k in self.kernel.shape[1:] for p in (k // 2, k // 2))

        spatial_shape = x.shape[-n_dim:]
        batch_shape = x.shape[1:-n_dim]

        x = x.reshape(x.shape[0], -1, *spatial_shape).movedim(0, 1)
        y = conv_nd(x, self.kernel.unsqueeze(1).conj(), dim=tuple(range(-n_dim, 0)), transposed=True)
        y = adjoint_pad_nd(y, pad=pad, dims=tuple(range(-n_dim, 0)), mode=self.pad_mode).squeeze(1)
        y = y.reshape(*batch_shape, *spatial_shape)

        return (y,)
