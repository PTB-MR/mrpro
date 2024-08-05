"""Cartesian trajectory class."""

import torch
from einops import repeat

from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator


class KTrajectoryCartesian(KTrajectoryCalculator):
    """Cartesian trajectory."""

    def __call__(self, kheader: KHeader) -> KTrajectory:
        """Calculate Cartesian trajectory for given KHeader.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

        Returns
        -------
            Cartesian trajectory for given KHeader
        """
        # K-space locations along readout lines
        kx = self._kfreq(kheader)

        # Trajectory along phase and slice encoding
        ky = (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32)
        kz = (kheader.acq_info.idx.k2 - kheader.encoding_limits.k2.center).to(torch.float32)

        # Bring to correct dimensions
        ky = repeat(ky, '... k2 k1-> ... k2 k1 k0', k0=1)
        kz = repeat(kz, '... k2 k1-> ... k2 k1 k0', k0=1)
        return KTrajectory(kz, ky, kx)
