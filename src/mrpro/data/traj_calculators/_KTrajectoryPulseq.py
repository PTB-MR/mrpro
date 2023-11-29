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

from pathlib import Path

import numpy as np
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
        seq.read(file_path=self.seq_path)

        # calculate k-space trajectory using PyPulseq
        k_traj_adc, _, _, _, _ = seq.calculate_kspace()

        unique_idxs = {label: np.unique(getattr(kheader.acq_info.idx, label)) for label in ['k1', 'k2']}

        k1 = len(unique_idxs['k1'])
        k2 = len(unique_idxs['k2'])

        num_samples = kheader.acq_info.number_of_samples
        if len(torch.unique(num_samples)) > 1:
            raise ValueError('We  currently only support constant number of samples')

        # get number of samples as integer
        # ToDo: find more pythonic solution compatible with mypy
        k0 = int(num_samples[0].squeeze().tolist()[0])

        sample_size = num_samples.shape[0]

        k_traj_adc[0] = k_traj_adc[0] / np.max(np.abs(k_traj_adc[0])) * np.pi
        k_traj_adc[1] = k_traj_adc[1] / np.max(np.abs(k_traj_adc[1])) * np.pi
        k_traj_adc[2] = k_traj_adc[2] / np.max(np.abs(k_traj_adc[2])) * np.pi

        kx = torch.tensor(k_traj_adc[0]).view((sample_size, k2, k1, k0))
        ky = torch.tensor(k_traj_adc[1]).view((sample_size, k2, k1, k0))
        kz = torch.tensor(k_traj_adc[2]).view((sample_size, k2, k1, k0))

        # rearrange k-space trajectory to match MRpro convention
        kx = rearrange(
            kx,
            'other k2 k1 k0 -> (other k2 k1) k0',
            k0=k0,
            k2=k2,
            k1=k1,
            other=sample_size,
        )
        ky = rearrange(
            ky,
            'other k2 k1 k0 -> (other k2 k1) k0',
            k0=k0,
            k2=k2,
            k1=k1,
            other=sample_size,
        )
        kz = rearrange(
            kz,
            'other k2 k1 k0 -> (other k2 k1) k0',
            k0=k0,
            k2=k2,
            k1=k1,
            other=sample_size,
        )

        return KTrajectoryRawShape(kz, ky, kx)
