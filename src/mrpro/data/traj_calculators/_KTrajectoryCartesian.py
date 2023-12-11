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

from __future__ import annotations

import torch

from mrpro.data import KHeader
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators import KTrajectoryCalculator


class KTrajectoryCartesian(KTrajectoryCalculator):
    """Cartesian encoding."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(self, kheader: KHeader) -> KTrajectory:
        """Calculate cartesian phase encoding trajectory for given KHeader.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

        Returns
        -------
           cartesian phase encoding trajectory for given KHeader
        """

        # dkx, dky, dkz = 1 / kheader.recon_fov  # TODO: Check correct units
        kx_max = kheader.recon_fov.x / kheader.recon_matrix.x
        ky_max = kheader.recon_fov.y / kheader.recon_matrix.y
        kz_max = kheader.recon_fov.z / kheader.recon_matrix.z

        kx = torch.linspace(-kx_max, kx_max, kheader.recon_matrix.x)
        ky = torch.linspace(-ky_max, ky_max, kheader.recon_matrix.y)
        kz = torch.linspace(-kz_max, kz_max, kheader.recon_matrix.z)

        kx = kx[None, :, None, None]
        ky = ky[None, None, :, None]
        kz = kz[None, None, None, :]

        return KTrajectory(kz, ky, kx)
