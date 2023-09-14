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

from mrpro.data._KHeader import KHeader
from mrpro.data.traj_calculators._KTrajectoryRpe import KTrajectoryRpe


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

    def _apply_sunflower_shift_between_rpe_lines(self, krad: torch.Tensor, kang: torch.Tensor, kheader: KHeader):
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

    def calc_traj(self, kheader: KHeader) -> torch.Tensor:
        """Calculate sunflower golden angle RPE trajectory for given KHeader.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data.

        Returns
        -------
            sunflower golden angle radial phase encoding trajectory for given KHeader
        """
        # Calculate points along readout
        k0 = self._k0_traj(kheader.acq_info.number_of_samples, kheader.acq_info.center_sample)

        # Angles of phase encoding lines
        kang = (kheader.acq_info.idx.k2 * self.angle) % torch.pi

        # K-space locations along phase encoding lines
        krad = (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32)
        krad = self._apply_sunflower_shift_between_rpe_lines(krad, kang, kheader)
        krad *= 2 * torch.pi / kheader.encoding_limits.k1.max

        return self._combine_to_3d_traj(krad, kang, k0)
