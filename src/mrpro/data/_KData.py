"""Data class for MR raw data."""

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

import os
from pathlib import Path
from typing import Any

import ismrmrd
import numpy as np
import torch

from mrpro.data import KHeader
from mrpro.data.raw import bitmask_flag_to_strings
from mrpro.data.traj import KTrajectory


class KData:
    def __init__(
        self, header: KHeader, data: torch.Tensor, traj: torch.Tensor
    ) -> None:
        self.header: KHeader = header
        self._data: torch.Tensor = data
        self._traj: torch.Tensor = traj

    @classmethod
    def from_file(
        cls, filename: str | Path, ktrajectory_calculator: KTrajectory
    ) -> KData:
        # Check file is valid
        if not os.path.isfile(filename):
            print('%s is not a valid file' % filename)
            raise SystemExit

        # Read header
        dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
        hdr_xml = dset.read_xml_header()
        hdr = ismrmrd.xsd.CreateFromDocument(hdr_xml)
        dset.close()

        # Read k-space data
        with ismrmrd.File(filename) as mrd:
            acqs = mrd['dataset'].acquisitions[:]

        # Get indices for imaging data
        im_idx = []
        unique_acq_flags = set()
        for idx, acq in enumerate(acqs):
            for el in bitmask_flag_to_strings(acq.flags):
                unique_acq_flags.add(el)
            if 'ACQ_IS_NOISE_MEASUREMENT' not in bitmask_flag_to_strings(
                acq.flags
            ):
                im_idx.append(idx)

        kheader = KHeader()
        kheader.from_ismrmrd_header(hdr, len(im_idx))

        # Get k-space data
        kdata = torch.zeros(
            (
                len(im_idx),
                kheader.num_coils,
                acqs[im_idx[0]].number_of_samples,
            ),
            dtype=torch.complex64,
        )
        for idx in range(len(im_idx)):
            acq = acqs[im_idx[idx]]
            kdata[idx, :, :] = torch.tensor(acq.data, dtype=torch.complex64)
            # TODO: Make this faster
            kheader.acq_info.from_ismrmrd_acq_header(idx, acq)

        # Calculate trajectory
        ktraj = ktrajectory_calculator.calc_traj(kheader)

        # TODO: Check for partial Fourier and reflected readouts

        # Sort k-space data into (dim4, ncoils, k2, k1, k0)
        kdim_labels = (
            'kspace_encode_step_1',
            'kspace_encode_step_2',
            'average',
            'slice',
            'contrast',
            'phase',
            'repetition',
            'set',
        )
        kdim_num = np.asarray(
            [
                len(np.unique(getattr(kheader.acq_info, acq_label)))
                for acq_label in kdim_labels
            ]
        )

        # Ensure each dim4 covers the same number of k2 and k1 points
        for idx, acq_label in enumerate(kdim_labels[2:]):
            label_values = np.unique(getattr(kheader.acq_info, acq_label))
            for ind in range(len(label_values)):
                cidx_curr_label = tuple(
                    np.where(
                        getattr(kheader.acq_info, acq_label)
                        == label_values[ind]
                    )[0]
                )
                kdim_label_k1 = len(
                    np.unique(
                        kheader.acq_info.kspace_encode_step_1[cidx_curr_label]
                    )
                )
                kdim_label_k2 = len(
                    np.unique(
                        kheader.acq_info.kspace_encode_step_2[cidx_curr_label]
                    )
                )
                assert (
                    kdim_label_k1 == kdim_num[0]
                ), f"""{acq_label} has
                {kdim_label_k1} k1 points instead of {kdim_num[0]}"""
                assert (
                    kdim_label_k2 == kdim_num[1]
                ), f"""{acq_label} has
                {kdim_label_k2} k2 points instead of {kdim_num[1]}"""

        sort_ki = np.stack(
            (
                kheader.acq_info.kspace_encode_step_1,
                kheader.acq_info.kspace_encode_step_2,
                kheader.acq_info.average,
                kheader.acq_info.slice,
                kheader.acq_info.contrast,
                kheader.acq_info.phase,
                kheader.acq_info.repetition,
                kheader.acq_info.set,
            ),
            axis=0,
        )
        sort_idx = np.lexsort(sort_ki)

        new_shape = (
            np.prod(kdim_num[2:]),
            kdim_num[1],
            kdim_num[0],
        )
        kdata = torch.reshape(
            kdata[sort_idx, :, :], new_shape + kdata.shape[1:]
        )
        kdata = torch.moveaxis(kdata, (0, 1, 2, 3, 4), (0, 2, 3, 1, 4))

        ktraj = torch.reshape(
            ktraj[sort_idx, :, :], new_shape + ktraj.shape[1:]
        )
        ktraj = torch.moveaxis(ktraj, (0, 1, 2, 3, 4), (0, 2, 3, 1, 4))

        for slot in kheader.acq_info.__slots__:
            curr_attr = getattr(kheader.acq_info, slot)
            # TODO: Check for correct dimensionality in test function!
            if curr_attr.ndim == 2:
                curr_shape: Any = new_shape + (curr_attr.shape[1],)
            else:
                curr_shape = new_shape
            setattr(
                kheader.acq_info,
                slot,
                torch.reshape(curr_attr[sort_idx, ...], curr_shape),
            )

        return cls(kheader, kdata, ktraj)

    @property
    def traj(self) -> torch.Tensor:
        return self._traj

    @traj.setter
    def traj(self, value: torch.Tensor):
        self._traj = value

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        self._data = value
