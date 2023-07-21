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
from dataclasses import fields

import ismrmrd
import torch


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
        acquisitions: list of ismrmrd acquisistions to read from
        """

        def get_tensor(name, acqs):
            """Stacks the attribute from each acq in acqs."""

            if name in dir(acqs[0].idx):
                # If the attribute s in idx, we get it from there
                def get_attribute(acq): return getattr(acq.idx, name)
            else:
                # Otherwise we get it from the acquisition itself
                def get_attribute(acq): return getattr(acq, name)

            values = map(get_attribute, acqs)
            return torch.tensor(values)

        attributes = {field.name: get_tensor(field.name, acquisitions) for field in fields(cls)}
        return cls(**attributes)
