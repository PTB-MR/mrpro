"""KTrajectoryRawShape dataclass."""

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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange

from mrpro.data import KTrajectory


@dataclass(slots=True, frozen=True)
class KTrajectoryRawShape:
    """K-space trajectory shaped ((other*k2*k1),k0).

    Order of directions is always kz, ky, kx
    Shape of each of kx,ky,kz is ((other,k2,k1),k0) this means that e.g. slices, averages... have not yet been
    separated from the phase and slice encoding dimensions. The trajectory is in the same shape as the raw data in the
    raw data file.
    """

    kz: torch.Tensor
    """(other,k2,k1,k0), phase encoding direction k2 if Cartesian."""

    ky: torch.Tensor
    """(other,k2,k1,k0), phase encoding direction k1 if Cartesian."""

    kx: torch.Tensor
    """(other,k2,k1,k0), frequency encoding direction k0 if Cartesian."""

    repeat_detection_tolerance: None | float = 1e-3
    """tolerance for repeat detection, by default 1e-3, None to disable."""

    def reshape(
        self,
        sort_idx: np.ndarray,
        n_k2: int,
        n_k1: int,
    ) -> KTrajectory:
        """Resort and reshape the raw trajectory to KTrajectory.

        Parameters
        ----------
        sort_idx
            Index which defines how combined dimension (other k2 k1) needs to be sorted such that it can be separated
            into three separate dimensions using simple reshape operation. This information needs to be provided from
            kheader.acq_info.
        n_k2
            number of k2 points.
        n_k1
            number of k1 points.

        Returns
        -------
            KTrajectory with kx, ky and kz each in the shape (other k2 k1 k0).
        """
        # Resort and reshape
        kz = rearrange(self.kz[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)
        ky = rearrange(self.ky[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)
        kx = rearrange(self.kx[sort_idx, ...], '(other k2 k1) k0 -> other k2 k1 k0', k1=n_k1, k2=n_k2)

        return KTrajectory(kz, ky, kx, repeat_detection_tolerance=self.repeat_detection_tolerance)
