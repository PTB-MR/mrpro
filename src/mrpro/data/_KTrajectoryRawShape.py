"""KTrajectoryRawShape dataclass."""

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

from dataclasses import dataclass

import numpy as np
import torch

from mrpro.data import KTrajectory
from mrpro.utils import remove_repeat

from einops import rearrange


@dataclass(slots=True, init=False)
class KTrajectoryRawShape:
    """K-space trajectory in the shape of the original raw data ((other*k2*k1),k0).

    Order of directions is always kz, ky, kx
    Shape of each of kx,ky,kz is ((other,k2,k1),k0) this means that e.g. slices, averages... have not yet been
    separated from the phase and slice encoding dimensions. The trajectory is in the same shape as the raw data in the
    raw data file.
    """

    kz: torch.Tensor  # ((other,k2,k1),k0) #phase encoding direction, k2 if Cartesian
    ky: torch.Tensor  # ((other,k2,k1),k0) #phase encoding direction, k1 if Cartesian
    kx: torch.Tensor  # ((other,k2,k1),k0) #frequency encoding direction, k0 if Cartesian

    def __init__(
        self, kz: torch.Tensor, ky: torch.Tensor, kx: torch.Tensor):
        """Dataclass for the k-Space trajectory in the shape of the data in the raw data file.

        Parameters
        ----------
        kz, ky, kx:
            Trajectory coordinates to set
        """
        self.kz = kz
        self.ky = ky
        self.kx = kx


    def reshape(self, sort_idx: torch.Tensor, num_k2: int, num_k1: int) -> KTrajectory:
        """Resort and reshape the raw trajectory to the shape and order of KTrajectory.

        Parameters
        ----------
        sort_idx
            Index which defines how combined dimension (other k2 k1) is separated into three separate dimensions.
            This information needs to be provided from kheader.acq_info.
        num_k2
            Number of k2 points.
        num_k1
            Number of k1 points.

        Returns
        -------
            Ktrajectory with kx, ky and kz each in the shape (other k2 k1 k0).
        """

        # Resort and reshape
        kz = rearrange(self.kz[sort_idx,...], '(other k2 k1) k0 -> other k2 k1 k0', k1=num_k1, k2=num_k2)
        ky = rearrange(self.ky[sort_idx,...], '(other k2 k1) k0 -> other k2 k1 k0', k1=num_k1, k2=num_k2)
        kx = rearrange(self.kx[sort_idx,...], '(other k2 k1) k0 -> other k2 k1 k0', k1=num_k1, k2=num_k2)

        return KTrajectory(kz, ky, kx)

