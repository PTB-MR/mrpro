"""Patch extraction (sliding window) operator."""

from collections.abc import Sequence

import torch

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils.sliding_window import sliding_window


class PatchOp(LinearOperator):
    """Extract N-dimensional patches using a sliding window view.

    The adjoint assembles patches to an image.
    """

    def __init__(
        self,
        dim: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        stride: Sequence[int] | int | None = None,
        dilation: Sequence[int] | int = 1,
        domain_size: int | Sequence[int] | None = None,
        flatten_patches: bool = True,
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
        flatten_patches
            If True, flatten the leading grid dimensions to a single patch dimension.
            If False, keep shape ``(*grid_size, ...)`` for the forward output.
        """
        super().__init__()
        self.dim = (dim,) if isinstance(dim, int) else dim

        if len(set(self.dim)) != len(self.dim):
            raise ValueError('Axis indices must be unique')

        def check(param: int | Sequence[int], name: str) -> tuple[int, ...]:
            if isinstance(param, int):
                param = (param,) * len(self.dim)
            elif len(param) != len(self.dim):
                raise ValueError(f'Length mismatch: {name} must have length {len(self.dim)}')
            else:
                param = tuple(param)
            if any(val <= 0 for val in param):
                raise ValueError(f'{name} must be positive')
            return param

        self.patch_size = check(patch_size, 'patch_size')
        self.stride = check(stride, 'stride') if stride is not None else self.patch_size
        self.dilation = check(dilation, 'dilation')
        self.domain_size = check(domain_size, 'domain_size') if domain_size is not None else None
        self.flatten_patches = flatten_patches

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Extract N-dimensional patches from an input tensor using a sliding window.

        Parameters
        ----------
        x
            Input tensor from which to extract patches.

        Returns
        -------
            A tensor containing the extracted patches. The first dimension
            represents the number of patches, followed by the original
            tensor dimensions (excluding those used for patching), and then
            the patch dimensions themselves.
            Shape: `(n_patches, ... , patch_size_dim1, patch_size_dim2, ...)`.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of PatchOp.

        .. note::
            Prefer calling the instance of the PatchOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
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
        if self.flatten_patches:
            patches = patches.flatten(start_dim=0, end_dim=len(self.dim) - 1)
        return (patches,)

    def _adjoint_fast(self, patches: torch.Tensor) -> torch.Tensor:
        """Adjoint via reshape/permute for non-overlapping patches."""
        assert self.domain_size is not None  # mypy  # noqa: S101
        grid = tuple(s // p for s, p in zip(self.domain_size, self.patch_size, strict=True))
        n_dim = len(grid)
        if self.flatten_patches:
            patches = patches.unflatten(0, grid)
        permutation: list[int] = []
        reshape: list[int] = []
        dim = [d % (patches.ndim - n_dim) for d in self.dim]
        for i, size in enumerate(patches.shape[n_dim:]):
            if i in dim:
                j = dim.index(i)
                permutation.extend([j, n_dim + i])
                reshape.append(grid[j] * self.patch_size[j])
            else:
                permutation.append(n_dim + i)
                reshape.append(size)
        return patches.permute(*permutation).reshape(reshape)

    def _adjoint_scatter(self, patches: torch.Tensor) -> torch.Tensor:
        """Adjoint via scatter for overlapping patches."""
        assert self.domain_size is not None  # mypy  # noqa: S101
        k = len(self.dim)
        if not self.flatten_patches:
            patches = patches.flatten(start_dim=0, end_dim=k - 1)
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
        ).flatten(start_dim=0, end_dim=k - 1)
        if windowed_indices.shape[0] != patches.shape[0]:
            raise ValueError(
                f'Number of patches {patches.shape[0]} does not match the number of '
                f'expected patches {windowed_indices.shape[0]}'
            )

        assembled = patches.new_zeros(output_shape.numel())
        assembled.scatter_add_(dim=0, index=windowed_indices.flatten(), src=patches.flatten())
        assembled = assembled.reshape(output_shape)
        return assembled

    def adjoint(
        self,
        patches: torch.Tensor,
    ) -> tuple[torch.Tensor,]:
        """Assemble patches back into an image (adjoint operation).

        This method reconstructs an image by summing the provided patches
        at their respective locations, effectively reversing the patch
        extraction process. Overlapping areas are summed.

        Parameters
        ----------
        patches
            Tensor of patches to be assembled. Expected shape is
            `(n_patches, ..., patch_size_dim1, patch_size_dim2, ...)`.

        Returns
        -------
            The assembled image. Its shape will match the original image
            from which patches would have been extracted, with patch dimensions
            replaced by the original domain sizes along those dimensions.
        """
        if self.domain_size is None:
            raise ValueError('Domain size is not set. Please call forward first or set it at initialization.')
        if (
            self.stride == self.patch_size  # no overlap
            and all(d == 1 for d in self.dilation)  # no dilation
            and all(s % p == 0 for s, p in zip(self.domain_size, self.patch_size, strict=True))  # divisible
        ):
            return (self._adjoint_fast(patches),)
        else:
            return (self._adjoint_scatter(patches),)
