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
        self._nufft_directions: Sequence[int] = [direction_dict[d] for d in direction]

        if len(direction) != len(set(self._nufft_directions)):
            raise ValueError(f'Directions must be unique. Normalized directions are {self._nufft_directions}')

        if len(self._nufft_directions):
            nufft_traj = [
                ks
                for ks, i in zip((traj.kz, traj.ky, traj.kx), (-3, -2, -1), strict=True)
                if i in self._nufft_directions
            ]

            # Find out along which dimensions (k0, k1 or k2) nufft needs to be applied, i.e. where it is not singleton
            self._nufft_dims = []
            for dim in (-3, -2, -1):
                for ks in nufft_traj:
                    if ks.shape[dim] > 1:
                        self._nufft_dims.append(dim)
                        break  # one case where nufft is needed is enough for each dimension

            # For e.g. single shot acquisitions the number of dimensions do not necessarily match the number of
            # directions. This leads to a mismatch between reconstructed and expected dimensions. To avoid this we try
            # to find the most logical solution, i.e. add another singleton direction
            if len(self._nufft_directions) > len(self._nufft_dims):
                for dim in (-1, -2, -3):
                    if dim not in self._nufft_dims and ks.shape[dim] == 1:
                        self._nufft_dims.append(dim)
                    if len(self._nufft_directions) == len(self._nufft_dims):
                        break

            if isinstance(recon_matrix, SpatialDimension):
                im_size: Sequence[int] = [int(astuple(recon_matrix)[d]) for d in self._nufft_directions]
            else:
                if (n_recon_matrix := len(recon_matrix)) != (n_nufft_dir := len(self._nufft_directions)):
                    raise ValueError(f'recon_matrix should have {n_nufft_dir} entries but has {n_recon_matrix}')
                im_size = recon_matrix

            if isinstance(encoding_matrix, SpatialDimension):
                k_size: Sequence[int] = [int(astuple(encoding_matrix)[d]) for d in self._nufft_directions]
            else:
                if (n_enc_matrix := len(encoding_matrix)) != (n_nufft_dir := len(self._nufft_directions)):
                    raise ValueError(f'encoding_matrix should have {n_nufft_dir} entries but has {n_enc_matrix}')
                k_size = encoding_matrix

            grid_size = [int(size * nufft_oversampling) for size in im_size]
            omega_list = [
                k * 2 * torch.pi / ks
                for k, ks in zip(
                    nufft_traj,
                    k_size,
                    strict=True,
                )
            ]

            # Broadcast shapes not always needed but also does not hurt
            omega_list = [k.expand(*np.broadcast_shapes(*[k.shape for k in omega_list])) for k in omega_list]
            omega = torch.stack(omega_list, dim=-4)  # use the 'coil' dim for the direction
            self._traj_broadcast_shape = omega.shape

            keep_dims_210 = [-4, *self._nufft_dims]  # -4 is always coil
            permute_210 = [i for i in range(-omega.ndim, 0) if i not in keep_dims_210] + keep_dims_210
            # omega should be (sep_dims, 1, 2 or 3, nufft_dimensions)
            omega = omega.permute(*permute_210)
            omega = omega.flatten(end_dim=-len(keep_dims_210) - 1).flatten(start_dim=-len(keep_dims_210) + 1)

            # non-Cartesian -> Cartesian
            numpoints = [min(size, nufft_numpoints) for size in im_size]
            adj_nufft_op = KbNufftAdjoint(
                im_size=im_size,
                grid_size=grid_size,
                numpoints=numpoints,
                kbwidth=nufft_kbwidth,
            )
            self._nufft_type1 = lambda x: adj_nufft_op(x, omega, norm='ortho')

            # Cartesian -> non-Cartesian
            nufft_op = KbNufft(
                im_size=im_size,
                grid_size=grid_size,
                numpoints=numpoints,
                kbwidth=nufft_kbwidth,
            )
            self._nufft_type2 = lambda x: nufft_op(x, omega, norm='ortho')

            # we want to rearrange everything into (sep_dims)(joint_dims)(nufft_dims) where sep_dims are dimension
            # where the traj changes, joint_dims are dimensions where the traj does not change and nufft_dims are the
            # dimensions along which the nufft is applied. We have to do this for the (z-y-x) and (k2-k1-k0) space
            # separately. If we know two of the three dimensions we can infer the rest. We cannot do the other
            # dimensions here because they might be different between data and trajectory.
            self._joint_dims_210 = [
                d for d in [-3, -2, -1] if d not in self._nufft_dims and self._traj_broadcast_shape[d] == 1
            ]
            self._joint_dims_210.append(-4)  # -4 is always coil and always a joint dimension
            # TODO self._traj_broadcast_shape[d] is wrong...
            self._joint_dims_zyx = [
                d for d in [-3, -2, -1] if d not in self._nufft_directions and self._traj_broadcast_shape[d] == 1
            ]
            self._joint_dims_zyx.append(-4)  # -4 is always coil and always a joint dimension

    def _separate_joint_dimensions(
        self, data_ndim: int
    ) -> tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
        """Get the separate and joint dimensions for the current data.

        Parameters
        ----------
        data_ndim
            number of dimensions of the data

        Returns
        -------
            ((sep dims along zyx), (permute for zyx), (sep dims along 210), (permute for 210))

        """
        # We did most in _init_ and here we only have to check the other dimensions.
        joint_dims_other = []
        for d in range(-data_ndim, -4):
            if abs(d) > len(self._traj_broadcast_shape) or self._traj_broadcast_shape[d] == 1:
                joint_dims_other.append(d)

        sep_dims_xyz = [
            d
            for d in range(-data_ndim, 0)
            if d not in [*joint_dims_other, *self._joint_dims_zyx, *self._nufft_directions]
        ]
        permute_xyz = [*sep_dims_xyz, *joint_dims_other, *self._joint_dims_zyx, *self._nufft_directions]
        sep_dims_210 = [
            d for d in range(-data_ndim, 0) if d not in [*joint_dims_other, *self._joint_dims_210, *self._nufft_dims]
        ]
        permute_210 = [*sep_dims_210, *joint_dims_other, *self._joint_dims_210, *self._nufft_dims]
        return sep_dims_xyz, permute_xyz, sep_dims_210, permute_210

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
        if len(self._nufft_directions):
            # We rearrange x into (sep_dims, joint_dims, nufft_directions)
            sep_dims_xyz, permute_xyz, _, permute_210 = self._separate_joint_dimensions(x.ndim)
            unpermute_210 = np.argsort(permute_210)

            x = x.permute(*permute_xyz)
            unflatten_shape = x.shape[: -len(self._nufft_directions)]
            # combine sep_dims
            x = x.flatten(end_dim=len(sep_dims_xyz) - 1) if len(sep_dims_xyz) else x[None, :]
            # combine joint_dims
            x = x.flatten(start_dim=1, end_dim=-len(self._nufft_directions) - 1)

            x = self._nufft_type2(x)

            shape_nufft_dims = [self._traj_broadcast_shape[i] for i in self._nufft_dims]
            x = x.reshape(*unflatten_shape, *shape_nufft_dims)
            x = x.permute(*unpermute_210)
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
        if len(self._nufft_directions):
            # We rearrange x into (sep_dims, joint_dims, nufft_directions)
            _, permute_xyz, sep_dims_210, permute_210 = self._separate_joint_dimensions(x.ndim)
            unpermute_xyz = np.argsort(permute_xyz)

            x = x.permute(*permute_210)
            unflatten_other_shape = x.shape[: -len(self._nufft_dims) - 1]  # -1 for coil
            # combine sep_dims
            x = x.flatten(end_dim=len(sep_dims_210) - 1) if len(sep_dims_210) else x[None, :]
            # combine joint_dims and nufft_dims
            x = x.flatten(start_dim=1, end_dim=-len(self._nufft_dims) - 1).flatten(start_dim=2)

            x = self._nufft_type1(x)

            x = x.reshape(*unflatten_other_shape, -1, *x.shape[-len(self._nufft_directions) :])
            x = x.permute(*unpermute_xyz)
        return (x,)
