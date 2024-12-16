"""Acquisition information dataclass."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, overload

import ismrmrd
import numpy as np
import torch
from einops import rearrange
from typing_extensions import Self

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.unit_conversion import mm_to_m


def rearrange_acq_info_fields(field: object, pattern: str, **axes_lengths: dict[str, int]) -> object:
    """Change the shape of the fields in AcqInfo."""
    if isinstance(field, Rotation):
        return Rotation.from_matrix(rearrange(field.as_matrix(), pattern, **axes_lengths))

    if isinstance(field, torch.Tensor):
        return rearrange(field, pattern, **axes_lengths)

    return field


_convert_time_stamp_type: TypeAlias = Callable[
    [
        torch.Tensor,
        Literal[
            'acquisition_time_stamp', 'physiology_time_stamp_1', 'physiology_time_stamp_2', 'physiology_time_stamp_3'
        ],
    ],
    torch.Tensor,
]


def convert_time_stamp_siemens(
    timestamp: torch.Tensor,
    _: str,
) -> torch.Tensor:
    """Convert Siemens time stamp to seconds."""
    return timestamp.double() * 2.5e-3


@dataclass(slots=True)
class AcqIdx(MoveDataMixin):
    """Acquisition index for each readout."""

    k1: torch.Tensor
    """First phase encoding."""

    k2: torch.Tensor
    """Second phase encoding."""

    average: torch.Tensor
    """Signal average."""

    slice: torch.Tensor
    """Slice number (multi-slice 2D)."""

    contrast: torch.Tensor
    """Echo number in multi-echo."""

    phase: torch.Tensor
    """Cardiac phase."""

    repetition: torch.Tensor
    """Counter in repeated/dynamic acquisitions."""

    set: torch.Tensor
    """Sets of different preparation, e.g. flow encoding, diffusion weighting."""

    segment: torch.Tensor
    """Counter for segmented acquisitions."""

    user0: torch.Tensor
    """User index 0."""

    user1: torch.Tensor
    """User index 1."""

    user2: torch.Tensor
    """User index 2."""

    user3: torch.Tensor
    """User index 3."""

    user4: torch.Tensor
    """User index 4."""

    user5: torch.Tensor
    """User index 5."""

    user6: torch.Tensor
    """User index 6."""

    user7: torch.Tensor
    """User index 7."""


@dataclass(slots=True)
class UserValues(MoveDataMixin):
    """User Values used in AcqInfo."""

    float1: torch.Tensor
    float2: torch.Tensor
    float3: torch.Tensor
    float4: torch.Tensor
    float5: torch.Tensor
    float6: torch.Tensor
    float7: torch.Tensor
    float8: torch.Tensor
    int1: torch.Tensor
    int2: torch.Tensor
    int3: torch.Tensor
    int4: torch.Tensor
    int5: torch.Tensor
    int6: torch.Tensor
    int7: torch.Tensor
    int8: torch.Tensor


@dataclass(slots=True)
class PhysiologyTimestamps:
    """Time stamps relative to physiological triggering, e.g. ECG. Not in s but in vendor-specific time units."""

    timestamp1: torch.Tensor
    timestamp2: torch.Tensor
    timestamp3: torch.Tensor


@dataclass(slots=True)
class AcqInfo(MoveDataMixin):
    """Acquisition information for each readout."""

    idx: AcqIdx
    """Indices describing acquisitions (i.e. readouts)."""

    acquisition_time_stamp: torch.Tensor
    """Clock time stamp. Usually in seconds (Siemens: seconds since midnight)"""

    flags: torch.Tensor
    """A bit mask of common attributes applicable to individual acquisition readouts."""

    measurement_uid: torch.Tensor
    """Unique ID corresponding to the readout."""

    orientation: Rotation
    """Rotation describing the orientation of the readout, phase and slice encoding direction."""

    patient_table_position: SpatialDimension[torch.Tensor]
    """Offset position of the patient table, in LPS coordinates [m]."""

    physiology_time_stamps: PhysiologyTimestamps
    """Time stamps relative to physiological triggering, e.g. ECG. Not in s but in vendor-specific time units"""

    position: SpatialDimension[torch.Tensor]
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    sample_time_us: torch.Tensor
    """Readout bandwidth, as time between samples [us]."""

    scan_counter: torch.Tensor
    """Zero-indexed incrementing counter for readouts."""

    user: UserValues
    """User defined float or int values"""

    @overload
    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: Sequence[ismrmrd.acquisition.Acquisition],
        *,
        additional_fields: None,
        convert_time_stamp: _convert_time_stamp_type = convert_time_stamp_siemens,
    ) -> Self: ...

    @overload
    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: Sequence[ismrmrd.acquisition.Acquisition],
        *,
        additional_fields: Sequence[str],
        convert_time_stamp: _convert_time_stamp_type = convert_time_stamp_siemens,
    ) -> tuple[Self, tuple[torch.Tensor, ...]]: ...

    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: Sequence[ismrmrd.acquisition.Acquisition],
        *,
        additional_fields: Sequence[str] | None = None,
        convert_time_stamp: _convert_time_stamp_type = convert_time_stamp_siemens,
    ) -> Self | tuple[Self, tuple[torch.Tensor, ...]]:
        """Read the header of a list of acquisition and store information.

        Parameters
        ----------
        acquisitions
            list of ismrmrd acquisistions to read from. Needs at least one acquisition.
        additional_fields
            if supplied, additional information from fields with these names will be extracted from the
            ismrmrd acquisitions and returned as tensors.
        convert_time_stamp
            function used to convert the raw time stamps to seconds.
        """
        # Idea: create array of structs, then a struct of arrays,
        # convert it into tensors to store in our dataclass.
        # TODO: there might be a faster way to do this.

        if len(acquisitions) == 0:
            raise ValueError('Acquisition list must not be empty.')

        # Creating the dtype first and casting to bytes
        # is a workaround for a bug in cpython causing a warning
        # if np.array(AcquisitionHeader) is called directly.
        # also, this needs to check the dtype only once.
        acquisition_head_dtype = np.dtype(ismrmrd.AcquisitionHeader)
        headers = np.frombuffer(
            np.array([memoryview(a._head).cast('B') for a in acquisitions]),
            dtype=acquisition_head_dtype,
        )

        idx = headers['idx']

        def tensor(data: np.ndarray) -> torch.Tensor:
            # we have to convert first as pytoch cant create tensors from np.uint16 arrays
            # we use int32 for uint16 and int64 for uint32 to fit largest values.
            match data.dtype:
                case np.uint16:
                    data = data.astype(np.int32)
                case np.uint32 | np.uint64:
                    data = data.astype(np.int64)
            # Remove any unnecessary dimensions
            return torch.tensor(np.squeeze(data))

        def tensor_2d(data: np.ndarray) -> torch.Tensor:
            # Convert tensor to torch dtypes and ensure it is atleast 2D
            data_tensor = tensor(data)
            # Ensure that data is (k1*k2*other, >=1)
            if data_tensor.ndim == 1:
                data_tensor = data_tensor[:, None]
            elif data_tensor.ndim == 0:
                data_tensor = data_tensor[None, None]
            return data_tensor

        def spatialdimension_2d(data: np.ndarray) -> SpatialDimension[torch.Tensor]:
            # Ensure spatial dimension is (k1*k2*other, 1, 3)
            if data.ndim != 2:
                raise ValueError('Spatial dimension is expected to be of shape (N,3)')
            data = data[:, None, :]
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
            user0=tensor(idx['user'][:, 0]),
            user1=tensor(idx['user'][:, 1]),
            user2=tensor(idx['user'][:, 2]),
            user3=tensor(idx['user'][:, 3]),
            user4=tensor(idx['user'][:, 4]),
            user5=tensor(idx['user'][:, 5]),
            user6=tensor(idx['user'][:, 6]),
            user7=tensor(idx['user'][:, 7]),
        )
        user = UserValues(
            tensor_2d(headers['user_float'][:, 0]),
            tensor_2d(headers['user_float'][:, 1]),
            tensor_2d(headers['user_float'][:, 2]),
            tensor_2d(headers['user_float'][:, 3]),
            tensor_2d(headers['user_float'][:, 4]),
            tensor_2d(headers['user_float'][:, 5]),
            tensor_2d(headers['user_float'][:, 6]),
            tensor_2d(headers['user_float'][:, 7]),
            tensor_2d(headers['user_int'][:, 0]),
            tensor_2d(headers['user_int'][:, 1]),
            tensor_2d(headers['user_int'][:, 2]),
            tensor_2d(headers['user_int'][:, 3]),
            tensor_2d(headers['user_int'][:, 4]),
            tensor_2d(headers['user_int'][:, 5]),
            tensor_2d(headers['user_int'][:, 6]),
            tensor_2d(headers['user_int'][:, 7]),
        )
        physiology_time_stamps = PhysiologyTimestamps(
            convert_time_stamp(tensor_2d(headers['physiology_time_stamp'][:, 0]), 'physiology_time_stamp_1'),
            convert_time_stamp(tensor_2d(headers['physiology_time_stamp'][:, 1]), 'physiology_time_stamp_2'),
            convert_time_stamp(tensor_2d(headers['physiology_time_stamp'][:, 2]), 'physiology_time_stamp_3'),
        )
        acq_info = cls(
            idx=acq_idx,
            acquisition_time_stamp=convert_time_stamp(
                tensor_2d(headers['acquisition_time_stamp']), 'acquisition_time_stamp'
            ),
            flags=tensor_2d(headers['flags']),
            measurement_uid=tensor_2d(headers['measurement_uid']),
            orientation=Rotation.from_directions(
                spatialdimension_2d(headers['slice_dir']),
                spatialdimension_2d(headers['phase_dir']),
                spatialdimension_2d(headers['read_dir']),
            ),
            patient_table_position=spatialdimension_2d(headers['patient_table_position']).apply_(mm_to_m),
            position=spatialdimension_2d(headers['position']).apply_(mm_to_m),
            sample_time_us=tensor_2d(headers['sample_time_us']),
            scan_counter=tensor_2d(headers['scan_counter']),
            user=user,
            physiology_time_stamps=physiology_time_stamps,
        )

        if additional_fields is None:
            return acq_info
        else:
            additional_values = tuple(tensor_2d(headers[field]) for field in additional_fields)
            return acq_info, additional_values
