"""PCA Compression Operator."""

import einops
import numpy as np
import torch

from mrpro.operators.LinearOperator import LinearOperator


class PCACompressionOp(LinearOperator):
    """PCA based compression operator."""

    def __init__(
        self,
        data: torch.Tensor,
        n_components: int,
        compression_dim: int = -4,
        separate_dims: tuple[int, ...] | None = None,
    ):
        """Construct a PCA based compression operator.

        Parameters
        ----------
        data
            Data to used to find the principal components.
        n_components
            Number of principal components to keep.
        compression_dim
            Dimension along which compression is applied. Default to coil dimension.
        separate_dims
            Dimensions which have different a compression matrix. None means one compression matrix for all dimensions
            except for the compression dimension
        """
        super().__init__()

        if separate_dims is None:
            # global compression matrix
            data = data.moveaxis(compression_dim, -1).reshape(-1, data.shape[compression_dim])
        else:
            # different compression matrices
            # reshape to (*separate dimensions, -1, compression_dim)
            separate_dims_normalized = [i % data.ndim for i in separate_dims]
            compression_dim_normalized = compression_dim % data.ndim
            if compression_dim_normalized in separate_dims_normalized:
                raise ValueError('compression dimension must not be in separate_dims')
            permute_order = (
                separate_dims_normalized
                + [i for i in range(data.ndim) if i != compression_dim_normalized and i not in separate_dims_normalized]
                + [compression_dim_normalized]
            )
            data = data.permute(permute_order)
            data = data.flatten(start_dim=len(separate_dims), end_dim=-2)  # keep separate dimensions andata coil

        data = data - data.mean(-1, keepdim=True)
        correlation = einops.einsum(data, data.conj(), '... i coil1, ... i coil2 -> ... coil1 coil2')
        _, _, v = torch.svd(correlation)
        self.register_buffer('_compression_matrix', v[..., :n_components, :].clone())
        self._separate_dims = separate_dims
        self._compression_dim = compression_dim

    @staticmethod
    def _applymatrix(
        data: torch.Tensor, matrix: torch.Tensor, compression_dim: int, separate_dims: None | tuple[int, ...]
    ) -> torch.Tensor:
        if separate_dims is None:
            # global compression matrix
            data = data.moveaxis(compression_dim, -1)
            data = einops.einsum(matrix, data, 'doubt din, ... din-> ... doubt')
            data = data.moveaxis(-1, compression_dim)
            return data

        else:
            # multiple compression matrices
            # we figure out the required permutation and reshaping here (and not in the __init__) to allow a different
            # number of dimensions in the data than used at initialization.
            # the separate_dimensions only have to be specified such that the refer to matching dimensions
            separate_dims_normalized = [i % data.ndim for i in separate_dims]
            compression_dim_normalized = compression_dim % data.ndim
            joint_dims = [
                i for i in range(data.ndim) if i not in (*separate_dims_normalized, compression_dim_normalized)
            ]
            permute_order = np.argsort([*separate_dims_normalized, *joint_dims])
            n_broadcast_dims = data.ndim - len(separate_dims_normalized) - 1  # -1 for compression dimension
            matrix = matrix.reshape(*matrix.shape[:-2], *([1] * n_broadcast_dims), *matrix.shape[-2:])
            matrix = matrix.permute(*permute_order, -2, -1)
            data = data.moveaxis(compression_dim_normalized, -1)
            data = (matrix @ data.unsqueeze(-1)).squeeze(-1)
            data = data.moveaxis(-1, compression_dim_normalized)
            return data

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the compression to the data.

        Parameters
        ----------
        data
            data to be compressed

        Returns
        -------
            compressed data
        """
        return (self._applymatrix(data, self._compression_matrix, self._compression_dim, self._separate_dims),)

    def adjoint(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint compression to the data.

        Parameters
        ----------
        data
            compressed data

        Returns
        -------
            expanded data
        """
        return (self._applymatrix(data, self._compression_matrix.mH, self._compression_dim, self._separate_dims),)
