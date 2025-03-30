"""Patch (sliding window) operator."""

from collections.abc import Sequence

import torch

from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils.sliding_window import sliding_window


class PatchOp(LinearOperator):
    """Extracts N-dimensional patches using a sliding window view."""

    def __init__(
        self,
        dim: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        stride: Sequence[int] | int | None = None,
        dilation: Sequence[int] | int = 1,
        domain_size: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize the PatchOp.

        Parameters
        ----------
        dim
            Dimension(s) to extract patches from.
        patch_size
            Size of patches (window_shape).
        stride
            Stride between patches. Set to `patch_size` if None.
        dilation
            Dilation factor of the patches
        domain_size
            Size of the domain in the dimnsions `dim`.
            If None, it is inferred from the input tensor on the first call.
            This is only used in the adjoint method.
        """
        super().__init__()
        self.dim = (dim,) if isinstance(dim, int) else dim
        self.patch_size = patch_size
        self.stride = stride
        self.dilation = dilation
        self.domain_size = (domain_size,) if isinstance(domain_size, int) else domain_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Extract patches.

        Parameters
        ----------
        x
            Input tensor to extract patches from.

        Returns
        -------
        Tensor with shape
        `(n_patches,... patch_size_1, ... patch_size_2, ...)`
        """
        domain_size = tuple(x.shape[dim] for dim in self.dim)
        if self.domain_size is None:
            self.domain_size = domain_size
        elif tuple(self.domain_size) != domain_size:
            raise ValueError(
                f'Domain size {self.domain_size} does not match input shape in dimensions {self.dim}: {domain_size}'
            )
        patches = sliding_window(
            x=x,
            window_shape=self.patch_size,
            dim=self.dim,
            stride=self.stride,
            dilation=self.dilation,
        )
        patches = patches.flatten(start_dim=0, end_dim=len(self.dim) - 1)
        return (patches,)

    def adjoint(
        self,
        patches: torch.Tensor,
    ) -> tuple[torch.Tensor,]:
        """Perform the adjoint operation, i.e. assmable the patches."""
        if self.domain_size is None:
            raise ValueError('Domain size is not set. Please call forward first or set it at initialization.')

        output_shape_ = list(patches.shape[1:])
        for dim, size in zip(self.dim, self.domain_size, strict=True):
            output_shape_[dim] = size
        output_shape = torch.Size(output_shape_)
        indices = torch.arange(output_shape.numel(), device=patches.device).reshape(output_shape_)
        windowed_indices = sliding_window(
            x=indices,
            window_shape=self.patch_size,
            dim=self.dim,
            stride=self.stride,
            dilation=self.dilation,
        ).flatten(start_dim=0, end_dim=len(self.dim) - 1)
        if windowed_indices.shape[0] != patches.shape[0]:
            raise ValueError(
                f'Number of patches {patches.shape[0]} does not match the number of '
                f'expected patches {windowed_indices.shape[0]}'
            )

        assembled = patches.new_zeros(output_shape.numel())
        assembled.scatter_add_(dim=0, index=windowed_indices.flatten(), src=patches.flatten())
        assembled = assembled.reshape(output_shape)
        return (assembled,)
