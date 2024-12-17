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

    def create_manually(self, n_spokes: int = 240, n_k0: int = 256, initial_angle: float = 0) -> KTrajectory:
        """Generate KTrajectory object without header information.

        Args:
            angle (float, optional): Rotation angle per subsequent spoke. Defaults to pi * 0.618034.
            n_spokes (int, optional): Number of radial spokes. Defaults to 240.
            n_k0 (int, optional): Number of k0 points along each spoke. Defaults to 256.
            initial_angle (float, optional): Initial rotation angle of the first spoke. Defaults to 0.

        Returns
        -------
            radial 2D trajectory
        """
        center_sample = n_k0 // 2

        # K-space locations along readout lines
        radial = torch.arange(0, n_k0, dtype=torch.float32) - center_sample
        spoke_idx = torch.arange(n_spokes)

        # Angles of readout lines
        kang = (spoke_idx * self.angle + initial_angle)[None, None, :, None]

        # K-space radial coordinates
        kx = radial * torch.cos(kang)
        ky = radial * torch.sin(kang)
        kz = torch.zeros(1, 1, 1, 1)

        return KTrajectory(kz, ky, kx)
