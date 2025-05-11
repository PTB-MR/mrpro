"""Cartesian trajectory class."""

from collections.abc import Sequence

import torch

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.utils import RandomGenerator
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
        seed: int = 0,
    ) -> KTrajectory:
        """
        Generate k-space Gaussian weighted variable density sampling.

        Parameters
        ----------
        encoding_matrix
            Encoded K-space size, must have ``encoding_matrix.z=1``.
            If a single integer, a square k-space is considered.
        acceleration
            Acceleration factor (undersampling rate).
        n_center
            Number of fully-sampled center lines to always include.
        fwhm_ratio
            Full-width at half-maximum of the Gaussian relative to encoding_matrix.y.
            Larger values approach uniform sampling. Set to infinity for uniform sampling.
        n_other
            Batch size(s). The trajectory is different for each batch sample.
        seed
            Random seed for reproducibility.


        Returns
        -------
            Cartesian trajectory.

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
            raise ValueError(f'Number of center lines ({n_center}) exceeds number of lines to keep ({n_keep}).')

        rng = RandomGenerator(seed)
        k1_center = n_k1 // 2
        center_start = k1_center - n_center // 2
        center_end = center_start + n_center
        k1_idx = rng.gaussian_variable_density_samples(
            (*n_other, n_keep), low=0, high=n_k1, fwhm=fwhm_ratio * n_k1, always_sample=range(center_start, center_end)
        )

        return cls()(
            n_k0=n_k0,
            k0_center=n_k0 // 2,
            k1_idx=k1_idx[..., None, None, :, None],
            k1_center=k1_center,
            k2_idx=torch.tensor(0),
            k2_center=0,
        )
