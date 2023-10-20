# %%
# Imports
import numpy as np
import torch

from mrpro.data import KHeader
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryCalculator


# %%
class KTrajectoryRadial2D(KTrajectoryCalculator):
    """Radial 2D trajectory.

    Parameters
    ----------
    angle
        angle in rad between two radial phase encoding lines
    """

    def __init__(
        self,
        angle: float = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2)),
    ) -> None:
        super().__init__()
        self.angle: float = angle

    def _krad(self, kheader: KHeader):
        """Calculate the k-space locations along the read-out encoding lines.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data

        Returns
        -------
            Trajectory along ONE readout

        Raises
        ------
        ValueError
            Number of samples have to be the same for each readout
        ValueError
            Center sample has to be the same for each readout
        """

        num_samples = kheader.acq_info.number_of_samples
        center_sample = kheader.acq_info.center_sample

        if len(torch.unique(num_samples)) > 1:
            raise ValueError(
                'Radial 2D trajectory can only be calculated if each acquisition has the same number of samples'
            )
        if len(torch.unique(center_sample)) > 1:
            raise ValueError(
                'Radial 2D trajectory can only be calculated if each acquisition has the same center sample'
            )

        # Calculate points along readout
        nk0 = int(num_samples[0, 0, 0])
        k0 = torch.linspace(0, nk0 - 1, nk0, dtype=torch.float32) - center_sample[0, 0, 0]
        k0 *= 2 * torch.pi / nk0
        return k0

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

        # K-space locations along phase encoding lines
        krad = self._krad(kheader)

        # Angles of phase encoding lines
        kang = kheader.acq_info.idx.k1 * self.angle

        # K-space cartesian coordinates
        kx = krad * torch.cos(kang)[..., None]
        ky = krad * torch.sin(kang)[..., None]
        kz = torch.zeros(1, 1, 1, 1)

        return KTrajectory(kz, ky, kx)
