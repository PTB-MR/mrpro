import einops
import numpy as np
import torch

from mrpro.data import Data
from mrpro.operators import LinearOperator


class CoilCompression(LinearOperator):
    """PCA based soft coil compression operator."""

    def __init__(self, data: Data | torch.Tensor, n_components, separate_dims: tuple[int, ...] | None = None):
        """Construct a PCA based soft coil compression operator.

        Parameters
        ----------
        data
            Data to used to find the principal components.
        n_components
            Number of principal components to keep.
        separate_dim, optional
            tuple of dimensions to construct separate compression matrices for.
            None means one global compression matrix for all (but the coil) dimensions.
        """
        super().__init__()
        if isinstance(data, Data):
            d = data.data
        else:
            d = data
        coil_dimension = d.ndim - 4
        if separate_dims is None:
            # global compression matrix
            d = d.moveaxis(coil_dimension, -1).reshape(-1, d.shape[coil_dimension])
        else:
            # reshape to (*separate dimensions, -1, coils)
            separate_dims_normalized = [d.ndim + i if i < 0 else i for i in separate_dims]
            if coil_dimension in separate_dims_normalized:
                raise ValueError('coil dimension must not be in separate_dims')
            permute = (
                separate_dims_normalized
                + [i for i in range(d.ndim) if i != coil_dimension and i not in separate_dims_normalized]
                + [coil_dimension]
            )
            d = d.permute(permute)
            d = d.flatten(start_dim=len(separate_dims), end_dim=-2)  # keep separate dimensions and coil

        d = d - d.mean(-1, keepdim=True)
        d = einops.einsum(d, d.conj(), '... i coil1, ... i coil2 -> ... coil1 coil2')
        _, _, v = torch.svd(d)
        self._compression_matrix = v[..., :n_components, :]
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
            separate_dims_normalized = [data.ndim + i if i < 0 else i for i in separate_dims]
            # we do it here, to allow a different number if dimensions in the data than used in __init__
            # the separate_dimensions only have to be specified such that the refer to the same dimensions
            permute = np.argsort(
                list(separate_dims_normalized)
                + [i for i in range(data.ndim) if i != coil_dimension and i not in separate_dims_normalized]
            )
            n_broadcast_dims = data.ndim - len(separate_dims_normalized) - 1  # -1 for coil
            # TODO: can be simplify this?
            matrix = matrix.reshape(*matrix.shape[:-2], *([1] * n_broadcast_dims), *matrix.shape[-2:])
            matrix = matrix.permute(*permute, -2, -1)
            data = data.moveaxis(coil_dimension, -1).unsqueeze(-1)
            data = matrix @ data
            data = data.squeeze(-1).moveaxis(-1, coil_dimension)
            return (data,)

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the compression to the data."""
        return self._applymatrix(data, self._compression_matrix, self._separate_dims)

    def adjoint(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint compression to the data."""
        return self._applymatrix(data, self._compression_matrix.mH, self._separate_dims)
