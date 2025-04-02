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
    ) -> None:
        """Construct a PCA based compression operator.

        The operator carries out an SVD followed by a threshold of the `n_components` largest values along the last
        dimension of a data with shape `(*other, joint_dim, compression_dim)`.
        A single SVD is carried out for everything along `joint_dim`. `other` are batch dimensions.

        Consider combining this operator with `~mrpro.operators.RearrangeOp` to make sure the data is
        in the correct shape before applying.

        Parameters
        ----------
        data
            Data of shape `(*other, joint_dim, compression_dim)` to be used to find the principal components.
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
        self._compression_matrix = v[..., :n_components, :].clone()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the compression to the data.

        Parameters
        ----------
        x
            data to be compressed of shape `(*other, joint_dim, compression_dim)`

        Returns
        -------
            compressed data of shape `(*other, joint_dim, n_components)`
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply PCACompressionOp.

        Use `operator.__call__`, i.e. call `operator()` instead.
        """
        try:
            result = (self._compression_matrix @ x.unsqueeze(-1)).squeeze(-1)
        except RuntimeError as e:
            raise RuntimeError(
                'Shape mismatch in adjoint Compression: '
                f'Matrix {tuple(self._compression_matrix.shape)} '
                f'cannot be multiplied with Data {tuple(x.shape)}.'
            ) from e
        return (result,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint compression to the data.

        Parameters
        ----------
        y
            compressed data of shape `(*other, joint_dim, n_components)`

        Returns
        -------
            expanded data of shape `(*other, joint_dim, compression_dim)`
        """
        try:
            result = (self._compression_matrix.mH @ y.unsqueeze(-1)).squeeze(-1)
        except RuntimeError as e:
            raise RuntimeError(
                'Shape mismatch in adjoint Compression: '
                f'Matrix^H {tuple(self._compression_matrix.mH.shape)} '
                f'cannot be multiplied with Data {tuple(y.shape)}.'
            ) from e
        return (result,)
