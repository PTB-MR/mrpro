"""2D radial trajectory class."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from mrpro.data._KHeader import KHeader
from mrpro.data._KTrajectory import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryCalculator


class KTrajectoryRadial2D(KTrajectoryCalculator):
    """Radial 2D trajectory.

    Parameters
    ----------
    angle
        angle in rad between two radial lines
    """

    def __init__(self, angle: float = torch.pi * 0.618034) -> None:
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
        kang = kheader.acq_info.idx.k1 * self.angle

        # K-space cartesian coordinates
        kx = krad * torch.cos(kang)[..., None]
        ky = krad * torch.sin(kang)[..., None]
        kz = torch.zeros(1, 1, 1, 1)

        return KTrajectory(kz, ky, kx)
