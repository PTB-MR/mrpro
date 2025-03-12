"""Acquisition information dataclass."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from typing import Literal, TypeAlias, overload

import ismrmrd
import numpy as np
import torch
from typing_extensions import Self

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.reshape import unsqueeze_at, unsqueeze_right
from mrpro.utils.typing import TorchIndexerType
from mrpro.utils.unit_conversion import m_to_mm, mm_to_m

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


def convert_time_stamp_osi2(
    timestamp: torch.Tensor,
    _: str,
) -> torch.Tensor:
    """Convert OSI2 time stamp to seconds."""
    return timestamp.double() * 1e-3


def _int_factory() -> torch.Tensor:
    return torch.zeros(1, 1, 1, 1, 1, dtype=torch.int64)


def _float_factory() -> torch.Tensor:
    return torch.zeros(1, 1, 1, 1, 1, dtype=torch.float)


def _position_factory() -> SpatialDimension[torch.Tensor]:
    return SpatialDimension(
        torch.zeros(1, 1, 1, 1, 1, dtype=torch.float),
        torch.zeros(1, 1, 1, 1, 1, dtype=torch.float),
        torch.zeros(1, 1, 1, 1, 1, dtype=torch.float),
    )


@dataclass(slots=True)
class AcqIdx(MoveDataMixin):
    """Acquisition index for each readout."""

    k1: torch.Tensor = field(default_factory=_int_factory)
    """First phase encoding."""

    k2: torch.Tensor = field(default_factory=_int_factory)
    """Second phase encoding."""

    average: torch.Tensor = field(default_factory=_int_factory)
    """Signal average."""

    slice: torch.Tensor = field(default_factory=_int_factory)
    """Slice number (multi-slice 2D)."""

    contrast: torch.Tensor = field(default_factory=_int_factory)
    """Echo number in multi-echo."""

    phase: torch.Tensor = field(default_factory=_int_factory)
    """Cardiac phase."""

    repetition: torch.Tensor = field(default_factory=_int_factory)
    """Counter in repeated/dynamic acquisitions."""

    set: torch.Tensor = field(default_factory=_int_factory)
    """Sets of different preparation, e.g. flow encoding, diffusion weighting."""

    segment: torch.Tensor = field(default_factory=_int_factory)
    """Counter for segmented acquisitions."""

    user0: torch.Tensor = field(default_factory=_int_factory)
    """User index 0."""

    user1: torch.Tensor = field(default_factory=_int_factory)
    """User index 1."""

    user2: torch.Tensor = field(default_factory=_int_factory)
    """User index 2."""

    user3: torch.Tensor = field(default_factory=_int_factory)
    """User index 3."""

    user4: torch.Tensor = field(default_factory=_int_factory)
    """User index 4."""

    user5: torch.Tensor = field(default_factory=_int_factory)
    """User index 5."""

    user6: torch.Tensor = field(default_factory=_int_factory)
    """User index 6."""

    user7: torch.Tensor = field(default_factory=_int_factory)
    """User index 7."""

    def __post_init__(self) -> None:
        """Ensure that all indices are broadcastable."""
        f = [getattr(self, field.name) for field in fields(self)]
        try:
            torch.broadcast_shapes(*[field.shape for field in f])
        except RuntimeError:
            raise ValueError('The acquisition index dimensions must be broadcastable.') from None
        if any(x.ndim < 5 for x in f):
            raise ValueError('The acquisition index tensors should each have at least 5 dimensions.')


@dataclass(slots=True)
class UserValues(MoveDataMixin):
    """User Values used in AcqInfo."""

    float0: torch.Tensor = field(default_factory=_float_factory)
    float1: torch.Tensor = field(default_factory=_float_factory)
    float2: torch.Tensor = field(default_factory=_float_factory)
    float3: torch.Tensor = field(default_factory=_float_factory)
    float4: torch.Tensor = field(default_factory=_float_factory)
    float5: torch.Tensor = field(default_factory=_float_factory)
    float6: torch.Tensor = field(default_factory=_float_factory)
    float7: torch.Tensor = field(default_factory=_float_factory)
    int0: torch.Tensor = field(default_factory=_int_factory)
    int1: torch.Tensor = field(default_factory=_int_factory)
    int2: torch.Tensor = field(default_factory=_int_factory)
    int3: torch.Tensor = field(default_factory=_int_factory)
    int4: torch.Tensor = field(default_factory=_int_factory)
    int5: torch.Tensor = field(default_factory=_int_factory)
    int6: torch.Tensor = field(default_factory=_int_factory)
    int7: torch.Tensor = field(default_factory=_int_factory)


@dataclass(slots=True)
class PhysiologyTimestamps(MoveDataMixin):
    """Time stamps relative to physiological triggering, e.g. ECG, in seconds."""

    timestamp0: torch.Tensor = field(default_factory=_float_factory)
    timestamp1: torch.Tensor = field(default_factory=_float_factory)
    timestamp2: torch.Tensor = field(default_factory=_float_factory)


@dataclass(slots=True)
class AcqInfo(MoveDataMixin):
    """Acquisition information for each readout."""

    idx: AcqIdx = field(default_factory=AcqIdx)
    """Indices describing acquisitions (i.e. readouts)."""

    acquisition_time_stamp: torch.Tensor = field(default_factory=_float_factory)
    """Clock time stamp [s] (Siemens: seconds since midnight)"""

    flags: torch.Tensor = field(default_factory=_int_factory)
    """A bit mask of common attributes applicable to individual acquisition readouts."""

    orientation: Rotation = field(default_factory=lambda: Rotation.identity((1, 1, 1, 1, 1)))
    """Rotation describing the orientation of the readout, phase and slice encoding direction."""

    patient_table_position: SpatialDimension[torch.Tensor] = field(default_factory=_position_factory)
    """Offset position of the patient table, in LPS coordinates [m]."""

    physiology_time_stamps: PhysiologyTimestamps = field(default_factory=PhysiologyTimestamps)
    """Time stamps relative to physiological triggering, e.g. ECG [s]."""

    position: SpatialDimension[torch.Tensor] = field(default_factory=_position_factory)
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    sample_time_us: torch.Tensor = field(default_factory=_float_factory)
    """Readout bandwidth, as time between samples [us]."""

    user: UserValues = field(default_factory=UserValues)
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

        def tensor_5d(data: np.ndarray) -> torch.Tensor:
            # Convert tensor to torch dtypes and ensure it is 5D
            data_tensor = tensor(data)
            return unsqueeze_right(data_tensor, 5 - data_tensor.ndim)

        def spatialdimension_5d(data: np.ndarray) -> SpatialDimension[torch.Tensor]:
            data_tensor = torch.tensor(data, dtype=torch.float32)
            data_tensor = unsqueeze_at(data_tensor, -2, 5 - data_tensor.ndim + 1)
            # all spatial dimensions are float32
            return SpatialDimension.from_array_xyz(data_tensor)

        acq_idx = AcqIdx(
            k1=tensor_5d(idx['kspace_encode_step_1']),
            k2=tensor_5d(idx['kspace_encode_step_2']),
            average=tensor_5d(idx['average']),
            slice=tensor_5d(idx['slice']),
            contrast=tensor_5d(idx['contrast']),
            phase=tensor_5d(idx['phase']),
            repetition=tensor_5d(idx['repetition']),
            set=tensor_5d(idx['set']),
            segment=tensor_5d(idx['segment']),
            user0=tensor_5d(idx['user'][:, 0]),
            user1=tensor_5d(idx['user'][:, 1]),
            user2=tensor_5d(idx['user'][:, 2]),
            user3=tensor_5d(idx['user'][:, 3]),
            user4=tensor_5d(idx['user'][:, 4]),
            user5=tensor_5d(idx['user'][:, 5]),
            user6=tensor_5d(idx['user'][:, 6]),
            user7=tensor_5d(idx['user'][:, 7]),
        )
        user = UserValues(
            tensor_5d(headers['user_float'][:, 0]),
            tensor_5d(headers['user_float'][:, 1]),
            tensor_5d(headers['user_float'][:, 2]),
            tensor_5d(headers['user_float'][:, 3]),
            tensor_5d(headers['user_float'][:, 4]),
            tensor_5d(headers['user_float'][:, 5]),
            tensor_5d(headers['user_float'][:, 6]),
            tensor_5d(headers['user_float'][:, 7]),
            tensor_5d(headers['user_int'][:, 0]),
            tensor_5d(headers['user_int'][:, 1]),
            tensor_5d(headers['user_int'][:, 2]),
            tensor_5d(headers['user_int'][:, 3]),
            tensor_5d(headers['user_int'][:, 4]),
            tensor_5d(headers['user_int'][:, 5]),
            tensor_5d(headers['user_int'][:, 6]),
            tensor_5d(headers['user_int'][:, 7]),
        )
        physiology_time_stamps = PhysiologyTimestamps(
            convert_time_stamp(tensor_5d(headers['physiology_time_stamp'][:, 0]), 'physiology_time_stamp_1'),
            convert_time_stamp(tensor_5d(headers['physiology_time_stamp'][:, 1]), 'physiology_time_stamp_2'),
            convert_time_stamp(tensor_5d(headers['physiology_time_stamp'][:, 2]), 'physiology_time_stamp_3'),
        )
        acq_info = cls(
            idx=acq_idx,
            acquisition_time_stamp=convert_time_stamp(
                tensor_5d(headers['acquisition_time_stamp']), 'acquisition_time_stamp'
            ),
            flags=tensor_5d(headers['flags']),
            orientation=Rotation.from_directions(
                spatialdimension_5d(headers['slice_dir']),
                spatialdimension_5d(headers['phase_dir']),
                spatialdimension_5d(headers['read_dir']),
            ),
            patient_table_position=spatialdimension_5d(headers['patient_table_position']).apply_(mm_to_m),
            position=spatialdimension_5d(headers['position']).apply_(mm_to_m),
            sample_time_us=tensor_5d(headers['sample_time_us']),
            user=user,
            physiology_time_stamps=physiology_time_stamps,
        )

        if additional_fields is None:
            return acq_info
        else:
            additional_values = tuple(tensor_5d(headers[field]) for field in additional_fields)
            return acq_info, additional_values

    def write_to_ismrmrd_acquisition(
        self, acquisition: ismrmrd.Acquisition, idx: TorchIndexerType
    ) -> ismrmrd.Acquisition:
        """Overwrite ISMRMRD acquisition information for single acquisition."""
        acquisition.idx.kspace_encode_step_1 = self.idx.k1[idx]
        acquisition.idx.kspace_encode_step_2 = self.idx.k2[idx]
        acquisition.idx.average = self.idx.average[idx]
        acquisition.idx.slice = self.idx.slice[idx]
        acquisition.idx.contrast = self.idx.contrast[idx]
        acquisition.idx.phase = self.idx.phase[idx]
        acquisition.idx.repetition = self.idx.repetition[idx]
        acquisition.idx.set = self.idx.set[idx]
        acquisition.idx.segment = self.idx.segment[idx]
        acquisition.idx.user = (
            self.idx.user0[idx],
            self.idx.user1[idx],
            self.idx.user2[idx],
            self.idx.user3[idx],
            self.idx.user4[idx],
            self.idx.user5[idx],
            self.idx.user6[idx],
            self.idx.user7[idx],
        )

        # active_channesl, number_of_samples and trajectory_dimensions are read-only and cannot be set
        acquisition.patient_table_position = self.patient_table_position[idx].apply(m_to_mm).zyx[::-1]  # zyx -> xyz
        directions = self.orientation[idx].as_directions()
        acquisition.slice_dir = directions[0].zyx[::-1]  # zyx -> xyz
        acquisition.phase_dir = directions[1].zyx[::-1]
        acquisition.read_dir = directions[2].zyx[::-1]
        acquisition.position = self.position[idx].apply(m_to_mm).zyx[::-1]
        acquisition.sample_time_us = self.sample_time_us[idx]
        acquisition.user_float = (
            self.user.float0[idx],
            self.user.float1[idx],
            self.user.float2[idx],
            self.user.float3[idx],
            self.user.float4[idx],
            self.user.float5[idx],
            self.user.float6[idx],
            self.user.float7[idx],
        )
        acquisition.user_int = (
            self.user.int0[idx],
            self.user.int1[idx],
            self.user.int2[idx],
            self.user.int3[idx],
            self.user.int4[idx],
            self.user.int5[idx],
            self.user.int6[idx],
            self.user.int7[idx],
        )
        # Time stamps are saved as Siemens time stamps in units of 2.5ms
        acquisition.acquisition_time_stamp = int(self.acquisition_time_stamp[idx] / 0.0025)
        acquisition.physiology_time_stamp = (
            int(self.physiology_time_stamps.timestamp0[idx] / 0.0025),
            int(self.physiology_time_stamps.timestamp1[idx] / 0.0025),
            int(self.physiology_time_stamps.timestamp2[idx] / 0.0025),
        )
        return acquisition
