"""2D radial trajectory class."""

import torch
from einops import repeat

from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator


class KTrajectoryRadial2D(KTrajectoryCalculator):
    """Radial 2D trajectory."""

    def __init__(self, angle: float = torch.pi * 0.618034) -> None:
        """Initialize KTrajectoryRadial2D.

        Parameters
        ----------
        angle
            angle in rad between two radial lines
        """
        super().__init__()
        self.angle: float = angle

    def __call__(self, kheader: KHeader) -> KTrajectory:
        """Calculate radial 2D trajectory for given KHeader.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

        Returns
        -------
            radial 2D trajectory for given KHeader
        """
        # K-space locations along readout lines
        krad = self._kfreq(kheader)

        # Angles of readout lines
        kang = repeat(kheader.acq_info.idx.k1 * self.angle, '... k2 k1 -> ... k2 k1 k0', k0=1)

        # K-space radial coordinates
        kx = krad * torch.cos(kang)
        ky = krad * torch.sin(kang)
        kz = torch.zeros(1, 1, 1, 1)

        return KTrajectory(kz, ky, kx)
