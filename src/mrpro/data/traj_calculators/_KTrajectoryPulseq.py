"""K-space trajectory from .seq file class."""

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

from pathlib import Path

import pypulseq as pp
import torch
from einops import rearrange

from mrpro.data import KHeader
from mrpro.data import KTrajectoryRawShape
from mrpro.data.traj_calculators import KTrajectoryCalculator


class KTrajectoryPulseq(KTrajectoryCalculator):
    """Trajectory from .seq file.

    Parameters
    ----------
    seq_path
        absolute path to .seq file
    """

    def __init__(self, seq_path: str | Path) -> None:
        super().__init__()
        self.seq_path = seq_path

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
        # create PyPulseq Sequence object and read .seq file
        seq = pp.Sequence()
        seq.read(file_path=str(self.seq_path))

        # calculate k-space trajectory using PyPulseq
        k_traj_adc_numpy, _, _, _, _ = seq.calculate_kspace()
        k_traj_adc = torch.from_numpy(k_traj_adc_numpy)

        unique_idxs = {label: torch.unique(getattr(kheader.acq_info.idx, label)) for label in ['k1', 'k2']}

        k1 = len(unique_idxs['k1'])
        k2 = len(unique_idxs['k2'])

        n_samples = kheader.acq_info.number_of_samples
        n_samples = torch.unique(n_samples)
        if len(n_samples) > 1:
            raise ValueError('We currently only support constant number of samples')
        k0 = int(n_samples.item())

        sample_size = n_samples.shape[0]

        def reshape_pulseq_traj(k_traj: torch.Tensor, encoding_size: int):
            k_traj *= encoding_size / (2 * torch.max(torch.abs(k_traj)))
            return rearrange(k_traj, '(other k2 k1 k0) -> (other k2 k1) k0', k0=k0, k2=k2, k1=k1, other=sample_size)

        # rearrange k-space trajectory to match MRpro convention
        kx = reshape_pulseq_traj(k_traj_adc[0], kheader.encoding_matrix.x)
        ky = reshape_pulseq_traj(k_traj_adc[1], kheader.encoding_matrix.y)
        kz = reshape_pulseq_traj(k_traj_adc[2], kheader.encoding_matrix.z)

        return KTrajectoryRawShape(kz, ky, kx)
