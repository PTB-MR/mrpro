"""MR raw data / k-space data class."""

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

import dataclasses
import datetime
from pathlib import Path

import h5py
import ismrmrd
import numpy as np
import torch
from einops import rearrange

from mrpro.data import AcqInfo
from mrpro.data import KHeader
from mrpro.data._EncodingLimits import Limits
from mrpro.data._KTrajectory import KTrajectory
from mrpro.data.enums import AcqFlags

KDIM_SORT_LABELS = (
    'kspace_encode_step_1',
    'kspace_encode_step_2',
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
)


class KData:
    def __init__(
        self,
        header: KHeader,
        data: torch.Tensor,
        traj: torch.Tensor,
    ) -> None:
        self._header: KHeader = header
        self._data: torch.Tensor = data
        self._traj: torch.Tensor = traj

    @classmethod
    def from_file(
        cls, filename: str | Path, ktrajectory: KTrajectory, header_overwrites: dict[str, object] | None = None
    ) -> KData:
        """Load k-space data from an ISMRMRD file.

        Parameters:
        ----------
            filename: Path to the ISMRMRD file
            ktrajectory: KTrajectory defining the trajectory to use # TODO: Maybe provide a default based on the header?
            header_overwrites: Dictionary of key-value pairs to overwrite the header
            dataset: Name of the dataset to load (siemens_to_ismrmrd creates dataset, dataset_1, dataset_2, ...)
        """
        # Can raise FileNotFoundError
        with ismrmrd.File(filename, 'r') as file:
            ds = file['dataset']
            ismrmrd_header = ds.header
            acquisitions = ds.acquisitions[:]
            try:
                mtime: int = h5py.h5g.get_objinfo(ds['data']._contents.id).mtime
            except AttributeError:
                mtime = 0
            modification_time = datetime.datetime.fromtimestamp(mtime)

        # Noise data must be handled separately
        acquisitions = list(filter(lambda acq: not (AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value & acq.flags), acquisitions))
        acqinfo = AcqInfo.from_ismrmrd_acquisitions(acquisitions)

        # Raises ValueError if required fields are missing in the header
        kheader = KHeader.from_ismrmrd(
            ismrmrd_header,
            acqinfo,
            defaults={
                'datetime': modification_time,  # use the modification time of the dataset as fallback
                'trajectory': ktrajectory,
            },
            overwrite=header_overwrites,
        )
        kdata = torch.stack([torch.as_tensor(acq.data, dtype=torch.complex64) for acq in acquisitions])

        # Fill k0 limits if they were set to zero / not set in the header
        if kheader.encoding_limits.kspace_encoding_step_0.length == 1:
            kheader.encoding_limits.kspace_encoding_step_0 = Limits(0, kdata.shape[-1] - 1, kdata.shape[-1] // 2)

        # TODO: Check for partial Fourier and reflected readouts

        # Sort kdata and acq_info into ("all other dim", coils, k2, k1, k0) / ("all other dim", k2, k1, acq_info_dims)
        # Fist, ensure each the non k1/k2 dimensions covers the same number of k1 and k2 points
        unique_idxs = {label: np.unique(getattr(kheader.acq_info, label)) for label in KDIM_SORT_LABELS}
        num_k1 = len(unique_idxs['kspace_encode_step_1'])
        num_k2 = len(unique_idxs['kspace_encode_step_2'])

        for label, idxs in unique_idxs.items():
            if label in ('kspace_encode_step_1', 'kspace_encode_step_2'):
                continue
            for idx in idxs:
                idx_matches_in_current_label = torch.nonzero(getattr(kheader.acq_info, label) == idx)
                current_num_k1 = len(torch.unique(kheader.acq_info.kspace_encode_step_1[idx_matches_in_current_label]))
                current_num_k2 = len(torch.unique(kheader.acq_info.kspace_encode_step_2[idx_matches_in_current_label]))
                if current_num_k1 != num_k1:
                    raise ValueError(f'Number of k1 points in {label}: {current_num_k1}. Expected: {num_k1}')
                if current_num_k2 != num_k2:
                    raise ValueError(f'Number of k2 points in {label}: {current_num_k2}. Expected: {num_k2}')
        # using np.lexsort as it looks a bit more familiar than looping and torch.argsort(..., stable=True)
        sort_ki = np.stack([getattr(kheader.acq_info, label) for label in KDIM_SORT_LABELS], axis=0)
        sort_idx = np.lexsort(sort_ki)

        kdata = rearrange(kdata[sort_idx], '(other k2 k1) coil k0 -> other coil k2 k1 k0', k1=num_k1, k2=num_k2)
        for field in dataclasses.fields(kheader.acq_info):
            current = getattr(kheader.acq_info, field.name)
            reshaped = rearrange(current[sort_idx], '(other k2 k1) ... -> other k2 k1 ...', k1=num_k1, k2=num_k2)
            setattr(kheader.acq_info, field.name, reshaped)

        # Calculate trajectory and check for shape mismatches
        ktraj = ktrajectory.calc_traj(kheader)
        if ktraj.shape[0] != 1 and ktraj.shape[0] != kdata.shape[0]:  # allow broadcasting in "other" dimensions
            raise ValueError(
                'shape mismatch between ktrajectory and kdata:\n'
                f'{ktraj.shape[0]} not broadcastable to {kdata.shape[0]}'
            )
        if ktraj.shape[2:] != kdata.shape[2:]:
            raise ValueError(
                'shape mismatch between kdata and ktrajectory in (k2, k1, k0) dimensions:\n'
                f'{ktraj.shape[2:]} != {kdata.shape[2:]}'
            )

        return cls(kheader, kdata, ktraj)

    @property
    def traj(self) -> torch.Tensor:
        return self._traj

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def header(self) -> KHeader:
        return self._header
