# %%
"""K-space trajectory from .seq file class."""

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
import pypulseq as pp
import torch
from einops import rearrange

from mrpro.data import KHeader
from mrpro.data._KTrajectoryRawShape import KTrajectoryRawShape
from mrpro.data.traj_calculators._KTrajectoryCalculator import KTrajectoryCalculator


class KTrajectorySeq(KTrajectoryCalculator):
    """Trajectory from .seq file.

    Parameters
    ----------
    path
        absolute path to .seq file
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def __call__(self, kheader: KHeader) -> KTrajectoryRawShape:
        """Calculate trajectory from given .seq file and header information.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

        Returns
        -------
            trajectory of type KTrajectoryRawShape
        """
        # k1 : num spirals
        # k2 : num slices
        # k0 : num k space points per spiral

        seq = pp.Sequence()
        seq.read(file_path=self.path)
        k_traj_adc, _, _, _, _ = seq.calculate_kspace()

        unique_idxs = {label: np.unique(getattr(kheader.acq_info.idx, label)) for label in ['k1', 'k2']}
        k1 = len(unique_idxs['k1'])
        k2 = len(unique_idxs['k2'])
        k0 = int(k_traj_adc.shape[1] / k1 / k2)

        k_traj_adc[0] = k_traj_adc[0] / np.max(np.abs(k_traj_adc[0])) * np.pi
        k_traj_adc[1] = k_traj_adc[1] / np.max(np.abs(k_traj_adc[1])) * np.pi
        k_traj_adc[2] = k_traj_adc[2] / np.max(np.abs(k_traj_adc[2])) * np.pi

        kx = torch.tensor(k_traj_adc[0]).view((1, k2, k1, k0))
        ky = torch.tensor(k_traj_adc[1]).view((1, k2, k1, k0))
        kz = torch.tensor(k_traj_adc[2]).view((1, k2, k1, k0))
        kx = rearrange(kx, 'other k2 k1 k0 -> (other k2 k1) k0', k2=k2, k1=k1, other=1)
        ky = rearrange(ky, 'other k2 k1 k0 -> (other k2 k1) k0', k2=k2, k1=k1, other=1)
        kz = rearrange(kz, 'other k2 k1 k0 -> (other k2 k1) k0', k2=k2, k1=k1, other=1)

        return KTrajectoryRawShape(kz, ky, kx)
