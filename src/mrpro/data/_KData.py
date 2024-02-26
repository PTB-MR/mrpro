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
from mrpro.data import KTrajectory
from mrpro.data import KTrajectoryRawShape
from mrpro.data import Limits
from mrpro.data.enums import AcqFlags
from mrpro.data.traj_calculators import KTrajectoryCalculator
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd

KDIM_SORT_LABELS = (
    'k1',
    'k2',
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
)

# Same criteria as https://github.com/wtclarke/pymapvbvd/blob/master/mapvbvd/mapVBVD.py uses
DEFAULT_IGNORE_FLAGS = (
    AcqFlags.ACQ_IS_NOISE_MEASUREMENT
    | AcqFlags.ACQ_IS_DUMMYSCAN_DATA
    | AcqFlags.ACQ_IS_HPFEEDBACK_DATA
    | AcqFlags.ACQ_IS_NAVIGATION_DATA
    | AcqFlags.ACQ_IS_PHASECORR_DATA
    | AcqFlags.ACQ_IS_PHASE_STABILIZATION
    | AcqFlags.ACQ_IS_PHASE_STABILIZATION_REFERENCE
    | AcqFlags.ACQ_IS_PARALLEL_CALIBRATION
)


@dataclasses.dataclass(slots=True, frozen=True)
class KData:
    """MR raw data / k-space data class."""

    header: KHeader
    data: torch.Tensor
    traj: KTrajectory

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        ktrajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd,
        header_overwrites: dict[str, object] | None = None,
        dataset_idx: int = -1,
        ignore_flags: AcqFlags = DEFAULT_IGNORE_FLAGS,
    ) -> KData:
        """Load k-space data from an ISMRMRD file.

        Parameters
        ----------
            filename
                path to the ISMRMRD file
            ktrajectory
                KTrajectoryCalculator to calculate the k-space trajectory or an already calculated KTrajectory
            header_overwrites
                dictionary of key-value pairs to overwrite the header
            dataset_idx
                index of the ISMRMRD dataset to load (converter creates dataset, dataset_1, ...), default is -1 (last)
            ignore_flags
                Acqisition flags to filter out. Defaults to all non-images as defined by pymapvbvd.
                Use ACQ_NO_FLAG to disable the filter.
                Note: If ACQ_IS_PARALLEL_CALIBRATION is set without also setting ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING
                      we interpret it as: Ignore if IS_PARALLEL_CALIBRATION and not PARALLEL_CALIBRATION_AND_IMAGING
        """

        # Can raise FileNotFoundError
        with ismrmrd.File(filename, 'r') as file:
            ds = file[list(file.keys())[dataset_idx]]
            ismrmrd_header = ds.header
            acquisitions = ds.acquisitions[:]
            try:
                mtime: int = h5py.h5g.get_objinfo(ds['data']._contents.id).mtime
            except AttributeError:
                mtime = 0
            modification_time = datetime.datetime.fromtimestamp(mtime)

        if (
            AcqFlags.ACQ_IS_PARALLEL_CALIBRATION in ignore_flags
            and AcqFlags.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING not in ignore_flags
        ):
            # if only ACQ_IS_PARALLEL_CALIBRATION is set, reinterpret it as: ignore if
            # ACQ_IS_PARALLEL_CALIBRATION is set and ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING is not set
            ignore_flags = ignore_flags & ~AcqFlags.ACQ_IS_PARALLEL_CALIBRATION
            acquisitions = list(
                filter(
                    lambda acq: (AcqFlags.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING.value & acq.flags)
                    or not (AcqFlags.ACQ_IS_PARALLEL_CALIBRATION.value & acq.flags),
                    acquisitions,
                )
            )

        acquisitions = list(filter(lambda acq: not (ignore_flags.value & acq.flags), acquisitions))

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

        # Create single kdata tensor from acquisition data
        kdata = torch.stack([torch.as_tensor(acq.data, dtype=torch.complex64) for acq in acquisitions])

        # Fill k0 limits if they were set to zero / not set in the header
        if kheader.encoding_limits.k0.length == 1:
            kheader.encoding_limits.k0 = Limits(0, kdata.shape[-1] - 1, kdata.shape[-1] // 2)

        # Sort kdata and acq_info into ("all other dim", coils, k2, k1, k0) / ("all other dim", k2, k1, acq_info_dims)
        # Fist, ensure each the non k1/k2 dimensions covers the same number of k1 and k2 points
        unique_idxs = {label: np.unique(getattr(kheader.acq_info.idx, label)) for label in KDIM_SORT_LABELS}

        # For reshaping into (other coils k2 k1 k0), the number of acqs must match the product of all unique_idxs
        num_total_unique = torch.as_tensor([len(unique_idxs[label]) for label in KDIM_SORT_LABELS]).prod()

        # Define function to find index label combinations. This is used to determine the dimensions of k1 and k2.
        def idx_label_combination(average_idx, slice_idx, contrast_idx, phase_idx, repetition_idx, set_idx):
            return torch.nonzero(
                (kheader.acq_info.idx.average == average_idx)
                & (kheader.acq_info.idx.slice == slice_idx)
                & (kheader.acq_info.idx.contrast == contrast_idx)
                & (kheader.acq_info.idx.phase == phase_idx)
                & (kheader.acq_info.idx.repetition == repetition_idx)
                & (kheader.acq_info.idx.set == set_idx)
            )

        idx_matches = idx_label_combination(
            *[unique_idxs[label][0] for label in KDIM_SORT_LABELS if label not in ('k1', 'k2')]
        )

        # Determine the number of k1 and k2 points
        if num_total_unique == kdata.shape[0]:
            # Data can be reshaped into (other coils k2 k1 k0))
            num_k1 = len(torch.unique(kheader.acq_info.idx.k1[idx_matches]))
            num_k2 = len(torch.unique(kheader.acq_info.idx.k2[idx_matches]))
        else:
            # Data is reshaped into (other 1 (k2 k1) k0)
            num_k1 = len(idx_matches)
            num_k2 = 1

        # Create all combinations of averages, slices, contrasts... to make the following loop easier to read
        idx_combinations = [
            [average_idx, slice_idx, contrast_idx, phase_idx, repetition_idx, set_idx]
            for average_idx in unique_idxs['average']
            for slice_idx in unique_idxs['slice']
            for contrast_idx in unique_idxs['contrast']
            for phase_idx in unique_idxs['phase']
            for repetition_idx in unique_idxs['repetition']
            for set_idx in unique_idxs['set']
        ]

        # Verify that k2 and k1 contain the same number of k-space points for all combinations of averages, slices...
        for average_idx, slice_idx, contrast_idx, phase_idx, repetition_idx, set_idx in idx_combinations:
            idx_matches = idx_label_combination(
                average_idx, slice_idx, contrast_idx, phase_idx, repetition_idx, set_idx
            )
            label_str = (
                f'[average {average_idx} | slice {slice_idx} | contrast {contrast_idx} | phase {phase_idx}'
                + f'| repetition {repetition_idx} | set {set_idx}]'
            )
            if num_total_unique == kdata.shape[0]:
                current_num_k1 = len(torch.unique(kheader.acq_info.idx.k1[idx_matches]))
                current_num_k2 = len(torch.unique(kheader.acq_info.idx.k2[idx_matches]))
                if current_num_k1 != num_k1:
                    raise ValueError(f'Number of k1 points in {label_str}: {current_num_k1}. Expected: {num_k1}')
                if current_num_k2 != num_k2:
                    raise ValueError(f'Number of k2 points in {label_str}: {current_num_k2}. Expected: {num_k2}')
            else:
                if len(idx_matches) != num_k1:
                    raise ValueError(f'Number of (k2 k1) points in {label_str}: {len(idx_matches)}. Expected: {num_k1}')

        # Sort the data according to the sorted indices
        sort_ki = np.stack([getattr(kheader.acq_info.idx, label) for label in KDIM_SORT_LABELS], axis=0)
        sort_idx = np.lexsort(sort_ki[:, :, 0])  # last dimension is 1
        kdata = rearrange(kdata[sort_idx], '(other k2 k1) coils k0 -> other coils k2 k1 k0', k1=num_k1, k2=num_k2)

        # Reshape the acquisition data and update the header acquisition infos accordingly
        def reshape_acq_data(data):
            return rearrange(data[sort_idx], '(other k2 k1) ... -> other k2 k1 ...', k1=num_k1, k2=num_k2)

        for field in dataclasses.fields(kheader.acq_info):
            current = getattr(kheader.acq_info, field.name)
            if isinstance(current, torch.Tensor):
                setattr(kheader.acq_info, field.name, reshape_acq_data(current))
            elif dataclasses.is_dataclass(current):
                for subfield in dataclasses.fields(current):
                    subcurrent = getattr(current, subfield.name)
                    setattr(current, subfield.name, reshape_acq_data(subcurrent))

        # Calculate trajectory and check if it matches the kdata shape
        match ktrajectory:
            case KTrajectoryIsmrmrd():
                ktraj = ktrajectory(acquisitions).reshape(sort_idx, num_k2, num_k1)
            case KTrajectoryCalculator():
                ktraj_calc = ktrajectory(kheader)
                if isinstance(ktraj_calc, KTrajectoryRawShape):
                    ktraj = ktraj_calc.reshape(sort_idx, num_k2, num_k1)
                else:
                    ktraj = ktraj_calc
            case KTrajectory():
                ktraj = ktrajectory
            case _:
                raise TypeError(
                    'ktrajectory must be KTrajectoryIsmrmrd, KTrajectory or KTrajectoryCalculator,'
                    f'not {type(ktrajectory)}'
                )

        try:
            shape = ktraj.broadcasted_shape
            torch.broadcast_shapes(kdata[..., 0, :, :, :].shape, shape)
        except RuntimeError:
            raise ValueError(
                f'Broadcasted shape trajectory do not match kdata: {shape} vs. {kdata.shape}. '
                'Please check the trajectory.'
            )

        return cls(kheader, kdata, ktraj)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: None | torch.dtype = None,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> KData:
        """Perform dtype and/or device conversion of trajectory and data.

        Parameters
        ----------
        device
            The destination device. Defaults to the current device.
        dtype
            Dtype of the k-space data, can only be torch.complex64 or torch.complex128.
            The dtype of the trajectory (torch.float32 or torch.float64) is then inferred from this.
        non_blocking
            If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        copy
            If True a new Tensor is created even when the Tensor already matches the desired conversion.
        memory_format
            The desired memory format of returned Tensor.
        """
        # Only complex64 and complex128 supported for kdata.
        # This will then lead to a trajectory of float32 and float64, respectively.
        if dtype is None:
            dtype_traj = None
        elif dtype == torch.complex64:
            dtype_traj = torch.float32
        elif dtype == torch.complex128:
            dtype_traj = torch.float64
        else:
            raise ValueError(f'dtype {dtype} not supported. Only torch.complex64 and torch.complex128 is supported.')

        return KData(
            header=self.header,
            data=self.data.to(
                device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, memory_format=memory_format
            ),
            traj=self.traj.to(
                device=device, dtype=dtype_traj, non_blocking=non_blocking, copy=copy, memory_format=memory_format
            ),
        )

    def cuda(
        self,
        device: torch.device | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> KData:
        """Create copy of object with trajectory and data in CUDA memory.

        Parameters
        ----------
        device
            The destination GPU device. Defaults to the current CUDA device.
        non_blocking
            If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        memory_format
            The desired memory format of returned Tensor.
        """
        return KData(
            header=self.header,
            data=self.data.cuda(
                device=device, non_blocking=non_blocking, memory_format=memory_format
            ),  # type: ignore [call-arg]
            traj=self.traj.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format),
        )

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> KData:
        """Create copy of object in CPU memory.

        Parameters
        ----------
        memory_format
            The desired memory format of returned Tensor.
        """
        return KData(
            header=self.header,
            data=self.data.cpu(memory_format=memory_format),  # type: ignore [call-arg]
            traj=self.traj.cpu(memory_format=memory_format),
        )
