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

from dataclasses import dataclass

import ismrmrd
import torch

from mrpro.data._utils import rgetattr


@dataclass(slots=True)
class AcqInfo:
    """Acquisiton Info Information about each readout."""

    acquisition_time_stamp: torch.Tensor
    active_channels: torch.Tensor
    available_channels: torch.Tensor
    average: torch.Tensor
    center_sample: torch.Tensor
    channel_mask: torch.Tensor
    contrast: torch.Tensor
    discard_post: torch.Tensor
    discard_pre: torch.Tensor
    encoding_space_ref: torch.Tensor
    flags: torch.Tensor
    kspace_encode_step_1: torch.Tensor
    kspace_encode_step_2: torch.Tensor
    measurement_uid: torch.Tensor
    number_of_samples: torch.Tensor
    patient_table_position: torch.Tensor
    phase: torch.Tensor
    phase_dir: torch.Tensor
    physiology_time_stamp: torch.Tensor
    position: torch.Tensor
    read_dir: torch.Tensor
    repetition: torch.Tensor
    sample_time_us: torch.Tensor
    scan_counter: torch.Tensor
    segment: torch.Tensor
    set: torch.Tensor
    slice: torch.Tensor
    user: torch.Tensor
    slice_dir: torch.Tensor
    trajectory_dimensions: torch.Tensor
    user_float: torch.Tensor
    user_int: torch.Tensor
    version: torch.Tensor

    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: list[ismrmrd.Acquisition],
    ) -> AcqInfo:
        """Reads the header of a list of acquisition and stores the
        information.

        Parameters:
        ----------
        acquisitions: list of ismrmrd acquisistions to read from. Needs at least one acquisition.
        """

        if len(acquisitions) == 0:
            raise ValueError('Acquisition list must not be empty.')

        def get_tensor(name) -> torch.Tensor:
            """
            Stacks the attribute from each acquisitions into a tensor.
            Parameters:
            ----------
            name: name of the attribute to stack. Will be resolved recursively,
                  e.g. 'idx.kspace_encode_step_1' will be resolved to acquisition.idx.kspace_encode_step_1
            """
            values = list(map(lambda acq: rgetattr(acq, name), acquisitions))
            return torch.tensor(values)

        attributes = dict(
            kspace_encode_step_1=get_tensor('idx.kspace_encode_step_1'),
            kspace_encode_step_2=get_tensor('idx.kspace_encode_step_2'),
            average=get_tensor('idx.average'),
            slice=get_tensor('idx.slice'),
            contrast=get_tensor('idx.contrast'),
            phase=get_tensor('idx.phase'),
            repetition=get_tensor('idx.repetition'),
            set=get_tensor('idx.set'),
            segment=get_tensor('idx.segment'),
            user=get_tensor('idx.user'),
            user_float=get_tensor('user_float'),
            user_int=get_tensor('user_int'),
            acquisition_time_stamp=get_tensor('acquisition_time_stamp'),
            flags=get_tensor('flags'),
            measurement_uid=get_tensor('measurement_uid'),
            scan_counter=get_tensor('scan_counter'),
            physiology_time_stamp=get_tensor('physiology_time_stamp'),
            active_channels=get_tensor('active_channels'),
            number_of_samples=get_tensor('number_of_samples'),
            available_channels=get_tensor('available_channels'),
            channel_mask=get_tensor('channel_mask'),
            discard_pre=get_tensor('discard_pre'),
            discard_post=get_tensor('discard_post'),
            center_sample=get_tensor('center_sample'),
            encoding_space_ref=get_tensor('encoding_space_ref'),
            trajectory_dimensions=get_tensor('trajectory_dimensions'),
            sample_time_us=get_tensor('sample_time_us'),
            position=get_tensor('position'),
            read_dir=get_tensor('read_dir'),
            phase_dir=get_tensor('phase_dir'),
            slice_dir=get_tensor('slice_dir'),
            patient_table_position=get_tensor('patient_table_position'),
            version=get_tensor('version'),
        )
        return cls(**attributes)
