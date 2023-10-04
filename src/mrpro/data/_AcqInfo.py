"""Acquisition information dataclass."""

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
import numpy as np
import torch

from mrpro.data import SpatialDimension


@dataclass(slots=True)
class AcqIdx:
    """Acquisition index for each readout."""

    k1: torch.Tensor
    k2: torch.Tensor
    average: torch.Tensor
    slice: torch.Tensor
    contrast: torch.Tensor
    phase: torch.Tensor
    repetition: torch.Tensor
    set: torch.Tensor
    segment: torch.Tensor
    user: torch.Tensor


@dataclass(slots=True)
class AcqInfo:
    """Acquisition information for each readout."""

    idx: AcqIdx
    acquisition_time_stamp: torch.Tensor
    active_channels: torch.Tensor
    available_channels: torch.Tensor
    center_sample: torch.Tensor
    channel_mask: torch.Tensor
    discard_post: torch.Tensor
    discard_pre: torch.Tensor
    encoding_space_ref: torch.Tensor
    flags: torch.Tensor
    measurement_uid: torch.Tensor
    number_of_samples: torch.Tensor
    patient_table_position: SpatialDimension[torch.Tensor]
    phase_dir: SpatialDimension[torch.Tensor]
    physiology_time_stamp: torch.Tensor
    position: SpatialDimension[torch.Tensor]
    read_dir: SpatialDimension[torch.Tensor]
    sample_time_us: torch.Tensor
    scan_counter: torch.Tensor
    slice_dir: SpatialDimension[torch.Tensor]
    trajectory_dimensions: torch.Tensor
    user_float: torch.Tensor
    user_int: torch.Tensor
    version: torch.Tensor

    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: list[ismrmrd.Acquisition],
    ) -> AcqInfo:
        """Read the header of a list of acquisition and store information.

        Parameters
        ----------
        acquisitions:
            list of ismrmrd acquisistions to read from. Needs at least one acquisition.
        """

        # Idea: create array of structs, then a struct of arrays,
        # convert it into tensors to store in our dataclass.
        # TODO: there might be a faster way to do this.

        if len(acquisitions) == 0:
            raise ValueError('Acquisition list must not be empty.')

        # Creating the dtype first and casting to bytes
        # is a workaround for a bug in cpython > 3.12 causing a warning
        # is np.array(AcquisitionHeader) is called directly.
        # also, this needs to check the dtyoe only once.
        acquisition_head_dtype = np.dtype(ismrmrd.AcquisitionHeader)
        headers = np.frombuffer(
            np.array([memoryview(a._head).cast('B') for a in acquisitions]), dtype=acquisition_head_dtype
        )

        idx = headers['idx']

        def tensor(data):
            # we have to convert first as pytoch cant create tensors from np.uint16 arrays
            # we use int32 for uint16 and int64 for uint32 to fit largest values.
            match data.dtype:
                case np.uint16:
                    data = data.astype(np.int32)
                case np.uint32 | np.uint64:
                    data = data.astype(np.int64)

            return torch.tensor(data).squeeze()

        def spatialdimension(data):
            # all spatial dimensions are float32
            return SpatialDimension[torch.Tensor].from_array_xyz(torch.tensor(data.astype(np.float32)))

        acq_idx = AcqIdx(
            k1=tensor(idx['kspace_encode_step_1']),
            k2=tensor(idx['kspace_encode_step_2']),
            average=tensor(idx['average']),
            slice=tensor(idx['slice']),
            contrast=tensor(idx['contrast']),
            phase=tensor(idx['phase']),
            repetition=tensor(idx['repetition']),
            set=tensor(idx['set']),
            segment=tensor(idx['segment']),
            user=tensor(idx['user']),
        )
        acq_info = cls(
            idx=acq_idx,
            acquisition_time_stamp=tensor(headers['acquisition_time_stamp']),
            active_channels=tensor(headers['active_channels']),
            available_channels=tensor(headers['available_channels']),
            center_sample=tensor(headers['center_sample']),
            channel_mask=tensor(headers['channel_mask']),
            discard_post=tensor(headers['discard_post']),
            discard_pre=tensor(headers['discard_pre']),
            encoding_space_ref=tensor(headers['encoding_space_ref']),
            flags=tensor(headers['flags']),
            measurement_uid=tensor(headers['measurement_uid']),
            number_of_samples=tensor(headers['number_of_samples']),
            patient_table_position=spatialdimension(headers['patient_table_position']),
            phase_dir=spatialdimension(headers['phase_dir']),
            physiology_time_stamp=tensor(headers['physiology_time_stamp']),
            position=spatialdimension(headers['position']),
            read_dir=spatialdimension(headers['read_dir']),
            sample_time_us=tensor(headers['sample_time_us']),
            scan_counter=tensor(headers['scan_counter']),
            slice_dir=spatialdimension(headers['slice_dir']),
            trajectory_dimensions=tensor(headers['trajectory_dimensions']),
            user_float=tensor(headers['user_float']),
            user_int=tensor(headers['user_int']),
            version=tensor(headers['version']),
        )
        return acq_info
