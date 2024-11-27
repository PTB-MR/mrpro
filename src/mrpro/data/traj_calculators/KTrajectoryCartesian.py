"""Cartesian trajectory class."""

import torch
from einops import repeat

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator


class KTrajectoryCartesian(KTrajectoryCalculator):
    """Cartesian trajectory."""

    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int,
        k1_idx: torch.Tensor,
        k1_center: int,
        k2_idx: torch.Tensor,
        k2_center: int,
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

        # Bring to correct dimensions
        ky = repeat(ky, '... k2 k1-> ... k2 k1 k0', k0=1)
        kz = repeat(kz, '... k2 k1-> ... k2 k1 k0', k0=1)
        return KTrajectory(kz, ky, kx)
