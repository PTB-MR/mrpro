"""Non-Uniform Fast Fourier Operator."""

from collections.abc import Sequence
from dataclasses import astuple
from typing import Literal

import numpy as np
import torch
from torchkbnufft import KbNufft, KbNufftAdjoint

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.LinearOperator import LinearOperator


class NonUniformFastFourierOp(LinearOperator, adjoint_as_backward=True):
    """Non-Uniform Fast Fourier Operator class."""

    def __init__(
        self,
        direction: Sequence[Literal['x', 'y', 'z', -3, -2, -1]],
        recon_matrix: SpatialDimension[int] | Sequence[int],
        encoding_matrix: SpatialDimension[int] | Sequence[int],
        traj: KTrajectory,
        nufft_oversampling: float = 2.0,
        nufft_numpoints: int = 6,
        nufft_kbwidth: float = 2.34,
    ) -> None:
        """Initialize Non-Uniform Fast Fourier Operator.

        Parameters
        ----------
        direction
            direction along which non-uniform FFT is applied
        recon_matrix
            Dimension of the reconstructed image. If this is SpatialDimension only values of directions will be used.
            Otherwise, it should be a Sequence of the same length as direction.
        encoding_matrix
            Dimension of the encoded k-space. If this is SpatialDimension only values of directions will be used.
            Otherwise, it should be a Sequence of the same length as direction.
        traj
            the k-space trajectories where the frequencies are sampled
        nufft_oversampling
            oversampling used for interpolation in non-uniform FFTs
        nufft_numpoints
            number of neighbors for interpolation in non-uniform FFTs
        nufft_kbwidth
            size of the Kaiser-Bessel kernel interpolation in non-uniform FFTs
        """
        super().__init__()

        # Convert to negative indexing
        direction_dict = {'z': -3, 'y': -2, 'x': -1, -3: -3, -2: -2, -1: -1}
        self._nufft_direction: Sequence[int] = [direction_dict[d] for d in direction]

        if len(self._nufft_direction) != len(set(self._nufft_direction)):
            raise ValueError(f'Directions must be unique. Normalized directions are {self._nufft_direction}')

        self._nufft_dims = self._nufft_direction
        if len(self._nufft_direction):
            if isinstance(recon_matrix, SpatialDimension):
                im_size: Sequence[int] = [int(astuple(recon_matrix)[d]) for d in self._nufft_direction]
            else:
                if (n_recon_matrix := len(recon_matrix)) != (n_nufft_dir := len(self._nufft_direction)):
                    raise ValueError(f'recon_matrix should have {n_nufft_dir} entries but has {n_recon_matrix}')
                im_size = recon_matrix

            if isinstance(encoding_matrix, SpatialDimension):
                k_size: Sequence[int] = [int(astuple(encoding_matrix)[d]) for d in self._nufft_direction]
            else:
                if (n_enc_matrix := len(encoding_matrix)) != (n_nufft_dir := len(self._nufft_direction)):
                    raise ValueError(f'encoding_matrix should have {n_nufft_dir} entries but has {n_enc_matrix}')
                k_size = encoding_matrix

            def get_traj(traj: KTrajectory, direction: Sequence[int]):
                return [k for k, i in zip((traj.kz, traj.ky, traj.kx), (-3, -2, -1), strict=True) if i in direction]

            grid_size = [int(size * nufft_oversampling) for size in im_size]
            omega = [
                k * 2 * torch.pi / ks
                for k, ks in zip(
                    get_traj(traj, self._nufft_direction),
                    k_size,
                    strict=True,
                )
            ]

            # Broadcast shapes not always needed but also does not hurt
            omega = [k.expand(*np.broadcast_shapes(*[k.shape for k in omega])) for k in omega]
            self.register_buffer('_omega', torch.stack(omega, dim=-4))  # use the 'coil' dim for the direction
            numpoints = [min(size, nufft_numpoints) for size in im_size]
            self._fwd_nufft_op = KbNufft(
                im_size=im_size,
                grid_size=grid_size,
                numpoints=numpoints,
                kbwidth=nufft_kbwidth,
            )
            self._adj_nufft_op = KbNufftAdjoint(
                im_size=im_size,
                grid_size=grid_size,
                numpoints=numpoints,
                kbwidth=nufft_kbwidth,
            )

            self._kshape = traj.broadcasted_shape
            self._im_size = im_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """NUFFT from image space to k-space.

        Parameters
        ----------
        x
            coil image data with shape: (... coils z y x)

        Returns
        -------
            coil k-space data with shape: (... coils k2 k1 k0)
        """
        if len(self._nufft_dims):
            # we need to move the nufft-dimensions to the end and flatten all other dimensions
            # so the new shape will be (... non_nufft_dims) coils nufft_dims
            # we could move the permute to __init__ but then we still would need to prepend if len(other)>1
            keep_dims = [-4, *self._nufft_dims]  # -4 is always coil
            permute = [i for i in range(-x.ndim, 0) if i not in keep_dims] + keep_dims
            unpermute = np.argsort(permute)

            x = x.permute(*permute)
            permuted_x_shape = x.shape
            x = x.flatten(end_dim=-len(keep_dims) - 1)

            # omega should be (... non_nufft_dims) n_nufft_dims (nufft_dims)
            # TODO: consider moving the broadcast along fft dimensions to __init__ (independent of x shape).
            omega = self._omega.permute(*permute)
            omega = omega.broadcast_to(*permuted_x_shape[: -len(keep_dims)], *omega.shape[-len(keep_dims) :])
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            x = self._fwd_nufft_op(x, omega, norm='ortho')

            shape_nufft_dims = [self._kshape[i] for i in self._nufft_dims]
            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], -1, *shape_nufft_dims)  # -1 is coils
            x = x.permute(*unpermute)
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """NUFFT from k-space to image space.

        Parameters
        ----------
        x
            coil k-space data with shape: (... coils k2 k1 k0)

        Returns
        -------
            coil image data with shape: (... coils z y x)
        """
        if len(self._nufft_dims):
            # we need to move the nufft-dimensions to the end, flatten them and flatten all other dimensions
            # so the new shape will be (... non_nufft_dims) coils (nufft_dims)
            keep_dims = [-4, *self._nufft_dims]  # -4 is coil
            permute = [i for i in range(-x.ndim, 0) if i not in keep_dims] + keep_dims
            unpermute = np.argsort(permute)

            x = x.permute(*permute)
            permuted_x_shape = x.shape
            x = x.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            omega = self._omega.permute(*permute)
            omega = omega.broadcast_to(*permuted_x_shape[: -len(keep_dims)], *omega.shape[-len(keep_dims) :])
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            x = self._adj_nufft_op(x, omega, norm='ortho')

            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], *x.shape[-len(keep_dims) :])
            x = x.permute(*unpermute)
        return (x,)
