"""PCA Compression Operator."""

import einops
import torch
from einops import repeat

from mrpro.operators.LinearOperator import LinearOperator


class PCACompressionOp(LinearOperator):
    """PCA based compression operator."""

    def __init__(
        self,
        data: torch.Tensor,
        n_components: int,
    ):
        """Construct a PCA based compression operator.

        The operator carries out an SVD followed by a threshold of the n_components largest values along the last
        dimension of a data with shape (*other, joint_dim, compression_dim). A single SVD is carried out for everything
        along joint_dim. Other are batch dimensions.

        Parameters
        ----------
        data
            Data to be used to find the principal components of shape (*other, joint_dim, compression_dim)
        n_components
            Number of principal components to keep along the compression_dim.
        """
        super().__init__()
        # different compression matrices along the *other dimensions
        data = data - data.mean(-1, keepdim=True)
        correlation = einops.einsum(data, data.conj(), '... joint comp1, ... joint comp2 -> ... comp1 comp2')
        _, _, v = torch.svd(correlation)
        # add joint_dim along which the the compression is the same
        v = repeat(v, '... comp1 comp2 -> ... joint_dim comp1 comp2', joint_dim=1)
        self.register_buffer('_compression_matrix', v[..., :n_components, :].clone())

    @staticmethod
    def _apply_matrix(data: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        if data.shape[-1] != matrix.shape[-1]:
            raise ValueError(f'Compression dimension does not match. Data: {data.shape[-1]} Matrix: {matrix.shape[-1]}')
        try:
            torch.broadcast_shapes(data.shape[:-1], matrix.shape[:-2])
        except RuntimeError:
            raise ValueError(
                f'Shape of matrix {matrix.shape[:-2]} cannot be croadcasted to data {data.shape[:-1]}'
            ) from None
        return (matrix @ data.unsqueeze(-1)).squeeze(-1)

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the compression to the data.

        Parameters
        ----------
        data
            data to be compressed of shape (*other, joint_dim, compression_dim)

        Returns
        -------
            compressed data of shape (*other, joint_dim, n_components)
        """
        return (self._apply_matrix(data, self._compression_matrix),)

    def adjoint(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint compression to the data.

        Parameters
        ----------
        data
            compressed data of shape (*other, joint_dim, n_components)

        Returns
        -------
            expanded data of shape (*other, joint_dim, compression_dim)
        """
        return (self._apply_matrix(data, self._compression_matrix.mH),)
