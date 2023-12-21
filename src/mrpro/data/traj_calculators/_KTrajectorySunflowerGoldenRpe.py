"""Radial phase encoding (RPE) trajectory class with sunflower pattern."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import numpy as np
import torch

from mrpro.data import KHeader
from mrpro.data.traj_calculators import KTrajectoryRpe


class KTrajectorySunflowerGoldenRpe(KTrajectoryRpe):
    """Radial phase encoding trajectory with a sunflower pattern.

    Parameters
    ----------
    rad_us_factor
        undersampling factor along radial phase encoding direction.
    """

    def __init__(
        self,
        rad_us_factor: float = 1.0,
    ) -> None:
        super().__init__(angle=torch.pi * 0.618034)
        self.rad_us_factor: float = rad_us_factor

    def _apply_sunflower_shift_between_rpe_lines(
        self, krad: torch.Tensor, kang: torch.Tensor, kheader: KHeader
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

        # Compensate for fov scaling
        num_rad_full = krad.shape[2] * self.rad_us_factor
        fov_scaling = (num_rad_full - self.rad_us_factor) / (num_rad_full - 1)
        return krad * fov_scaling

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
        return (kheader.acq_info.idx.k2 * self.angle) % torch.pi

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
        krad = (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32)
        krad = self._apply_sunflower_shift_between_rpe_lines(krad, kang, kheader)
        krad *= 2 * torch.pi / kheader.encoding_limits.k1.max
        return krad
