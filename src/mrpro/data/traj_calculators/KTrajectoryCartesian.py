"""Cartesian trajectory class."""

from collections.abc import Sequence

import torch

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.utils.reshape import unsqueeze_tensors_left


class KTrajectoryCartesian(KTrajectoryCalculator):
    """Cartesian trajectory."""

    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int | torch.Tensor,
        k1_idx: torch.Tensor,
        k1_center: int | torch.Tensor,
        k2_idx: torch.Tensor,
        k2_center: int | torch.Tensor,
        reversed_readout_mask: torch.Tensor | None = None,
        **_,
    ) -> KTrajectory:
        """Calculate Cartesian trajectory for given KHeader.

        Parameters
        ----------
        n_k0
            number of samples in k0
        k0_center
            position of k-space center in k0
        k1_idx
            indices of k1
        k1_center
            position of k-space center in k1
        k2_idx
            indices of k2
        k2_center
            position of k-space center in k2
        reversed_readout_mask
            boolean tensor indicating reversed readout

        Returns
        -------
            Cartesian trajectory for given KHeader
        """
        # K-space locations along readout lines
        kx = self._readout(n_k0, k0_center, reversed_readout_mask=reversed_readout_mask)

        # Trajectory along phase and slice encoding
        ky = (k1_idx - k1_center).to(torch.float32)
        kz = (k2_idx - k2_center).to(torch.float32)

        kz, ky, kx = unsqueeze_tensors_left(kz, ky, kx, ndim=5)
        return KTrajectory(kz, ky, kx)

    @classmethod
    def gaussian_variable_density(
        cls,
        encoding_matrix: SpatialDimension[int] | int,
        acceleration: float = 2.0,
        n_center: int = 10,
        fwhm_ratio: float = 0.3,
        n_other: Sequence[int] = (1,),
    ) -> KTrajectory:
        """
        Generate k-space Gaussian weighted variable density sampling.

        Parameters
        ----------
        encoding_matrix
            encoded K-space size, must have ``encoding_matrix.z=1``.
            If a single integer, a square k-space is considered.
        acceleration
            Acceleration factor (undersampling rate).
        n_center
            Number of fully-sampled center lines to always include.
        fwhm_ratio
            Full-width at half-maximum of the Gaussian relative to encoding_matrix.y.
            Larger values approach uniform sampling.

        Returns
        -------
            1D tensor of sorted selected k-space indices.

        Raises
        ------
        ValueError
            If `n_center` exceeds the total number of lines to keep given the acceleration.
        NotImplementedError
            If called with a 3D encoding matrix.
        """
        if isinstance(encoding_matrix, int):
            n_k1 = encoding_matrix
            n_k0 = encoding_matrix
        elif encoding_matrix.z > 1:
            raise NotImplementedError('Only 2D random trajectories can be created this way.')
        else:
            n_k1, n_k0 = encoding_matrix.y, encoding_matrix.x

        n_keep = min(int(n_k1 / acceleration), n_k1)
        if n_center > n_keep:
            raise ValueError(f'n_center ({n_center}) > total lines to keep ({n_keep})')

        k1_center = n_k1 // 2
        k1_center_start = k1_center - n_center // 2
        k1_center_end = k1_center_start + n_center
        if n_center:
            k1_idx = torch.arange(k1_center_start, k1_center_end).broadcast_to((*n_other, -1))

        if (n_rand := n_keep - n_center) > 0:
            sigma = fwhm_ratio / (2 * (2 * torch.log(torch.tensor(2.0))).sqrt())
            x = torch.linspace(-0.5, 0.5, n_k1)
            pdf = torch.exp(-(x**2) / (2 * sigma**2))
            pdf[k1_center_start:k1_center_end] = 0
            pdf = pdf.broadcast_to((*n_other, -1)).flatten(end_dim=-2)
            idx_rand = pdf.multinomial(n_rand, False)
            idx_rand = idx_rand.unflatten(0, n_other)
            if n_center:
                k1_idx = torch.cat([k1_idx, idx_rand], -1)
            else:
                k1_idx = idx_rand
            k1_idx = torch.sort(k1_idx)[0]

        return cls()(
            n_k0=n_k0,
            k0_center=n_k0 // 2,
            k1_idx=k1_idx[..., None, None, :, None],
            k1_center=k1_center,
            k2_idx=torch.tensor(0),
            k2_center=0,
        )
