"""Class for acquisition information of individual readouts."""

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

import ismrmrd
import torch


class AcqInfo:
    """Acquisiton Info Information about each readout."""

    __slots__ = (
        'acquisition_time_stamp',
        'active_channels',
        'available_channels',
        'average',
        'center_sample',
        'channel_mask',
        'contrast',
        'discard_post',
        'discard_pre',
        'encoding_space_ref',
        'flags',
        'kspace_encode_step_1',
        'kspace_encode_step_2',
        'measurement_uid',
        'number_of_samples',
        'patient_table_position',
        'phase',
        'phase_dir',
        'physiology_time_stamp',
        'position',
        'read_dir',
        'repetition',
        'sample_time_us',
        'scan_counter',
        'segment',
        'set',
        'slice',
        'slice_dir',
        'trajectory_dimensions',
        'user_float',
        'user_int',
        'version',
    )

    def __init__(self, num_acq: int) -> None:
        self.acquisition_time_stamp = torch.zeros((num_acq,), dtype=torch.int64)
        self.active_channels = torch.zeros((num_acq,), dtype=torch.int64)
        self.available_channels = torch.zeros((num_acq,), dtype=torch.int64)
        self.average = torch.zeros((num_acq,), dtype=torch.int64)
        self.center_sample = torch.zeros((num_acq,), dtype=torch.int64)
        self.channel_mask = torch.zeros((num_acq, ismrmrd.constants.CHANNEL_MASKS), dtype=torch.int64)
        self.contrast = torch.zeros((num_acq,), dtype=torch.int64)
        self.discard_post = torch.zeros((num_acq,), dtype=torch.int64)
        self.discard_pre = torch.zeros((num_acq,), dtype=torch.int64)
        self.encoding_space_ref = torch.zeros((num_acq,), dtype=torch.int64)
        self.flags = torch.zeros((num_acq,), dtype=torch.int64)
        self.kspace_encode_step_1 = torch.zeros((num_acq,), dtype=torch.int64)
        self.kspace_encode_step_2 = torch.zeros((num_acq,), dtype=torch.int64)
        self.measurement_uid = torch.zeros((num_acq,), dtype=torch.int64)
        self.number_of_samples = torch.zeros((num_acq,), dtype=torch.int64)
        self.patient_table_position = torch.zeros((num_acq, ismrmrd.constants.POSITION_LENGTH), dtype=torch.float32)
        self.phase = torch.zeros((num_acq,), dtype=torch.int64)
        self.phase_dir = torch.zeros((num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.physiology_time_stamp = torch.zeros((num_acq, ismrmrd.constants.PHYS_STAMPS), dtype=torch.float32)
        self.position = torch.zeros((num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.read_dir = torch.zeros((num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.repetition = torch.zeros((num_acq,), dtype=torch.int64)
        self.sample_time_us = torch.zeros((num_acq,), dtype=torch.int64)
        self.scan_counter = torch.zeros((num_acq,), dtype=torch.int64)
        self.segment = torch.zeros((num_acq,), dtype=torch.int64)
        self.set = torch.zeros((num_acq,), dtype=torch.int64)
        self.slice = torch.zeros((num_acq,), dtype=torch.int64)
        self.slice_dir = torch.zeros((num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.trajectory_dimensions = torch.zeros((num_acq,), dtype=torch.int64)
        self.user_float = torch.zeros((num_acq, ismrmrd.constants.USER_FLOATS), dtype=torch.float32)
        self.user_int = torch.zeros((num_acq, ismrmrd.constants.USER_INTS), dtype=torch.int64)
        self.version = torch.zeros((num_acq,), dtype=torch.int64)

    def read_ismrmrd_acq_header(
        self,
        to_index: int,
        acq: ismrmrd.Acquisition,
    ) -> None:
        """Reads the header of a single acquisition and stores the information
        at the given index.

        Parameters
        ----------
        to_index
            Index of the acquisition
        acq
            Acquisition to read
        """
        for slot in self.__slots__:
            curr_attr = getattr(self, slot)
            if slot in (
                'kspace_encode_step_1',
                'kspace_encode_step_2',
                'average',
                'slice',
                'contrast',
                'phase',
                'repetition',
                'set',
                'segment',
            ):
                curr_attr[to_index, ...] = torch.tensor(getattr(acq.idx, slot), dtype=curr_attr.dtype)

            else:
                curr_attr[to_index, ...] = torch.tensor(getattr(acq, slot), dtype=curr_attr.dtype)
            setattr(self, slot, curr_attr)
