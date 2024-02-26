"""Cartesian trajectory class."""

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

import torch
from einops import repeat

from mrpro.data import KHeader
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryCalculator


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
        ky = (kheader.acq_info.idx.k1[..., 0] - kheader.encoding_limits.k1.center).to(torch.float32)
        kz = (kheader.acq_info.idx.k2[..., 0] - kheader.encoding_limits.k2.center).to(torch.float32)

        # Bring to correct dimensions
        kx = repeat(kx, 'k0-> other k2 k1 k0', other=1, k2=1, k1=1)
        ky = repeat(ky, 'other k2 k1-> other k2 k1 k0', k0=1)
        kz = repeat(kz, 'other k2 k1-> other k2 k1 k0', k0=1)
        return KTrajectory(kz, ky, kx)
