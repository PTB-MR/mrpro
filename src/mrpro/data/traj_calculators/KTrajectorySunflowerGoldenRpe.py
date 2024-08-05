"""Radial phase encoding (RPE) trajectory class with sunflower pattern."""

import numpy as np
import torch
from einops import repeat

from mrpro.data.KHeader import KHeader
from mrpro.data.traj_calculators.KTrajectoryRpe import KTrajectoryRpe


class KTrajectorySunflowerGoldenRpe(KTrajectoryRpe):
    """Radial phase encoding trajectory with a sunflower pattern."""

    def __init__(self, rad_us_factor: float = 1.0) -> None:
        """Initialize KTrajectorySunflowerGoldenRpe.

        Parameters
        ----------
        rad_us_factor
            undersampling factor along radial phase encoding direction.
        """
        super().__init__(angle=torch.pi * 0.618034)
        self.rad_us_factor: float = rad_us_factor

    def _apply_sunflower_shift_between_rpe_lines(
        self,
        krad: torch.Tensor,
        kang: torch.Tensor,
        kheader: KHeader,
    ) -> torch.Tensor:
        """Shift radial phase encoding lines relative to each other.

        The shifts are applied to create a sunflower pattern of k-space points in the ky-kz phase encoding plane.
        The applied shifts can lead to a scaling of the FOV. This scaling depends on the undersampling factor along the
        radial phase encoding direction and is compensated for at the end.

        Parameters
        ----------
        krad
            k-space positions along each phase encoding line
        kang
            angles of the radial phase encoding lines
        kheader
            MR raw data header (KHeader) containing required meta data
        """
        kang = kang.flatten()
        _, indices = np.unique(kang, return_index=True)
        shift_idx = np.argsort(indices)

        # Apply sunflower shift
        golden_ratio = 0.5 * (np.sqrt(5) + 1)
        for ind, shift in enumerate(shift_idx):
            krad[kheader.acq_info.idx.k2 == ind] += ((shift * golden_ratio) % 1) - 0.5

        # Set asym k-space point to 0 because this point was used to obtain a self-navigator signal.
        krad[kheader.acq_info.idx.k1 == 0] = 0

        return krad

    def _kang(self, kheader: KHeader) -> torch.Tensor:
        """Calculate the angles of the phase encoding lines.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data

        Returns
        -------
            angles of phase encoding lines
        """
        return repeat((kheader.acq_info.idx.k2 * self.angle) % torch.pi, '... k2 k1 -> ... k2 k1 k0', k0=1)

    def _krad(self, kheader: KHeader) -> torch.Tensor:
        """Calculate the k-space locations along the phase encoding lines.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data

        Returns
        -------
            k-space locations along the phase encoding lines
        """
        kang = self._kang(kheader)
        krad = repeat(
            (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32),
            '... k2 k1 -> ... k2 k1 k0',
            k0=1,
        )
        krad = self._apply_sunflower_shift_between_rpe_lines(krad, kang, kheader)
        return krad
