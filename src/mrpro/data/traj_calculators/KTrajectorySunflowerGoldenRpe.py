"""Radial phase encoding (RPE) trajectory class with sunflower pattern."""

import numpy as np
import torch
from einops import repeat

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.utils.reshape import unsqueeze_tensors_left

GOLDEN_RATIO = 0.5 * (5**0.5 + 1)


class KTrajectorySunflowerGoldenRpe(KTrajectoryCalculator):
    """Radial phase encoding trajectory with a sunflower pattern."""

    def __init__(self) -> None:
        """Initialize KTrajectorySunflowerGoldenRpe."""
        self.angle = torch.pi * 0.618034

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

        Returns
        -------
            shifted radial positions
        """
        angles = angles.flatten()
        _, indices = np.unique(angles, return_index=True)
        shift_idx = np.argsort(indices)
        k2_idx, radial = torch.broadcast_tensors(k2_idx, radial)
        radial = radial.contiguous()
        for ind, shift in enumerate(shift_idx):
            radial[k2_idx == ind] += ((shift * GOLDEN_RATIO) % 1) - 0.5
        return radial

    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int | torch.Tensor,
        k1_idx: torch.Tensor,
        k1_center: int | torch.Tensor,
        k2_idx: torch.Tensor,
        reversed_readout_mask: torch.Tensor | None = None,
        **_,
    ) -> KTrajectory:
        """Calculate radial phase encoding trajectory for given header information.

        Parameters
        ----------
        n_k0
            number of samples in k0
        k0_center
            position of k-space center in k0
        k1_idx
            indices of k1 (radial)
        k1_center
            position of k-space center in k1
        k2_idx
            indices of k2 (angle)
        reversed_readout_mask
            boolean tensor indicating reversed readout

        Returns
        -------
            radial phase encoding trajectory for given KHeader
        """
        angles = repeat((k2_idx * self.angle) % torch.pi, '... k2 k1 -> ... k2 k1 k0', k0=1)

        radial = (k1_idx - k1_center).to(torch.float32)
        radial = self._apply_sunflower_shift_between_rpe_lines(radial, angles, k2_idx)
        # Asymmetric k-space point is used to obtain a self-navigator signal, thus should be in k-space center
        radial[(k1_idx == 0).broadcast_to(radial.shape)] = 0
        radial = repeat(radial, '... k2 k1 -> ... k2 k1 k0', k0=1)

        kz = radial * torch.sin(angles)
        ky = radial * torch.cos(angles)
        kx = self._readout(n_k0, k0_center, reversed_readout_mask=reversed_readout_mask)
        kz, ky, kx = unsqueeze_tensors_left(kz, ky, kx, ndim=5)

        return KTrajectory(kz, ky, kx)
