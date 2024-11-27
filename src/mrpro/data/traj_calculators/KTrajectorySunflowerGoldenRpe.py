"""Radial phase encoding (RPE) trajectory class with sunflower pattern."""

import numpy as np
import torch
from einops import repeat

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryCalculator

GOLDEN_RATIO = 0.5 * (5**0.5 + 1)


class KTrajectorySunflowerGoldenRpe(KTrajectoryCalculator):
    """Radial phase encoding trajectory with a sunflower pattern."""

    def __init__(self, radial_undersampling_factor: float = 1.0) -> None:
        """Initialize KTrajectorySunflowerGoldenRpe.

        Parameters
        ----------
        radial_undersampling_factor
            undersampling factor along radial phase encoding direction.
        """
        self.angle = torch.pi * 0.618034

        if radial_undersampling_factor != 1:
            raise NotImplementedError('Radial undersampling is not yet implemented')

    def _apply_sunflower_shift_between_rpe_lines(
        self,
        radial: torch.Tensor,
        angles: torch.Tensor,
        k2_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Shift radial phase encoding lines relative to each other.

        The shifts are applied to create a sunflower pattern of k-space points in the ky-kz phase encoding plane.

        Parameters
        ----------
        radial
            position along radial direction
        angles
            angle of spokes
        k2_idx
            indices in k2
        """
        angles = angles.flatten()
        _, indices = np.unique(angles, return_index=True)
        shift_idx = np.argsort(indices)
        for ind, shift in enumerate(shift_idx):
            radial[k2_idx == ind] += ((shift * GOLDEN_RATIO) % 1) - 0.5
        return radial

    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int,
        k1_idx: torch.Tensor,
        k1_center: int,
        k2_idx: torch.Tensor,
        reversed_readout_mask: torch.Tensor | None = None,
        **_,
    ) -> KTrajectory:
        """Calculate radial phase encoding trajectory for given KHeader.

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
        reversed_readout_mask
            boolean tensor indicating reversed readout

        Returns
        -------
            radial phase encoding trajectory for given KHeader
        """
        angles = repeat((k2_idx * self.angle) % torch.pi, '... k2 k1 -> ... k2 k1 k0', k0=1)
        radial = repeat((k1_idx - k1_center).to(torch.float32), '... k2 k1 -> ... k2 k1 k0', k0=1)
        radial = self._apply_sunflower_shift_between_rpe_lines(radial, angles, k2_idx)

        # Asymmetric k-space point is used to obtain a self-navigator signal, thus should be in k-space center
        radial[k1_idx == 0] = 0

        kz = radial * torch.sin(angles)
        ky = radial * torch.cos(angles)
        kx = self._readout(n_k0, k0_center, reversed_readout_mask=reversed_readout_mask)
        return KTrajectory(kz, ky, kx)
