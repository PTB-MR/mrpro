"""PCA Compression Operator."""

import einops
import numpy as np
import torch

from mrpro.data import Data
from mrpro.operators import LinearOperator


class PCACompressionOp(LinearOperator):
    """PCA based compression operator."""

    def __init__(self, data: torch.Tensor, n_components: int, compression_dim: int = -4, shared_dims: tuple[int, ...] | None = None):
        """Construct a PCA based compression operator.

        Parameters
        ----------
        data
            Data to used to find the principal components.
        n_components
            Number of principal components to keep.
        compression_dim
            Dimension along which compression is applied. Default to coil dimension.
        shared_dims
            Dimensions which share the same compression matrix.
        """
        super().__init__()

        coil_dimension = data.ndim - 4
        if separate_dims is None:
            # global compression matrix
            d = data.moveaxis(coil_dimension, -1).reshape(-1, data.shape[coil_dimension])
        else:
            # different compression matrices
            # reshape to (*separate dimensions, -1, coils)
            separate_dims_normalized = [i % data.ndim for i in separate_dims]
            if coil_dimension in separate_dims_normalized:
                raise ValueError('coil dimension must not be in separate_dims')
            permute_order = (
                separate_dims_normalized
                + [i for i in range(data.ndim) if i != coil_dimension and i not in separate_dims_normalized]
                + [coil_dimension]
            )
            d = data.permute(permute_order)
            d = data.flatten(start_dim=len(separate_dims), end_dim=-2)  # keep separate dimensions and coil

        d = d - data.mean(-1, keepdim=True)
        correlation = einops.einsum(d, data.conj(), '... i coil1, ... i coil2 -> ... coil1 coil2')
        _, _, v = torch.svd(correlation)
        self.register_buffer('_compression_matrix', v[..., :n_components, :].clone())
        self._separate_dims = separate_dims

    @staticmethod
    def _applymatrix(
        data: torch.Tensor, matrix: torch.Tensor, separate_dims: None | tuple[int, ...]
    ) -> tuple[torch.Tensor,]:
        coil_dimension = data.ndim - 4

        if separate_dims is None:
            # global compression matrix
            data = data.moveaxis(coil_dimension, -1)
            data = einops.einsum(matrix, data, 'out in, ... in-> ... out')
            data = data.moveaxis(-1, coil_dimension)
            return (data,)

        else:
            # multiple compression matrices
            # we figure out the required permutation and reshaping here (and not in the __init__) to allow a different
            # number of dimensions in the data than used in at initialization.
            # the separate_dimensions only have to be specified such that the refer to matching dimensions
            separate_dims_normalized = [i % data.ndim for i in separate_dims]
            joint_dims = [i for i in range(data.ndim) if i not in (*separate_dims_normalized, coil_dimension)]
            permute_order = np.argsort([*separate_dims_normalized, *joint_dims])
            n_broadcast_dims = data.ndim - len(separate_dims_normalized) - 1  # -1 for coil dimension
            matrix = matrix.reshape(*matrix.shape[:-2], *([1] * n_broadcast_dims), *matrix.shape[-2:])
            matrix = matrix.permute(*permute_order, -2, -1)
            data = data.moveaxis(coil_dimension, -1)
            data = (matrix @ data.unsqueeze(-1)).squeeze(-1)
            data = data.moveaxis(-1, coil_dimension)
            return (data,)

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
        return self._applymatrix(data, self._compression_matrix, self._separate_dims)

    def adjoint(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint compression to the data.

         Parameters
        ----------
        data
            compressed data

        Returns
        -------
            uncompressed data
        """
        return self._applymatrix(data, self._compression_matrix.mH, self._separate_dims)
