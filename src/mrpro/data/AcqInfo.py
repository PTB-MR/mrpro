"""Acquisition information dataclass."""

from collections.abc import Callable, Sequence
from dataclasses import field
from typing import Annotated, Literal, TypeAlias, overload

import ismrmrd
import numpy as np
import torch
from typing_extensions import Self

from mrpro.data.Dataclass import Dataclass
from mrpro.data.CheckDataMixin import Annotation, string_to_size
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.reshape import unsqueeze_at, unsqueeze_right
from mrpro.utils.unit_conversion import mm_to_m

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


IntTensor: TypeAlias = Annotated[
    torch.Tensor, Annotation(shape='*#other 1 #k2 #k1 1', dtype=(torch.int32, torch.int64))
]
FloatTensor: TypeAlias = Annotated[
    torch.Tensor, Annotation(shape='*#other 1 #k2 #k1 1', dtype=(torch.float32, torch.float64))
]
Float64Tensor: TypeAlias = Annotated[torch.Tensor, Annotation(shape='*#other 1 #k2 #k1 1', dtype=torch.float64)]
SpatialDimensionTensor: TypeAlias = Annotated[SpatialDimension[torch.Tensor], Annotation(shape='*#other 1 #k2 #k1 1')]
ShapeAnnotation = Annotation(shape='*#other 1 #k2 #k1 1')


class AcqIdx(Dataclass):
    """Acquisition index for each readout."""

    k1: IntTensor = field(default_factory=_int_factory)
    """First phase encoding."""

    k2: IntTensor = field(default_factory=_int_factory)
    """Second phase encoding."""

    average: IntTensor = field(default_factory=_int_factory)
    """Signal average."""

    slice: IntTensor = field(default_factory=_int_factory)
    """Slice number (multi-slice 2D)."""

    contrast: IntTensor = field(default_factory=_int_factory)
    """Echo number in multi-echo."""

    phase: IntTensor = field(default_factory=_int_factory)
    """Cardiac phase."""

    repetition: IntTensor = field(default_factory=_int_factory)
    """Counter in repeated/dynamic acquisitions."""

    set: IntTensor = field(default_factory=_int_factory)
    """Sets of different preparation, e.g. flow encoding, diffusion weighting."""

    segment: IntTensor = field(default_factory=_int_factory)
    """Counter for segmented acquisitions."""

    user0: IntTensor = field(default_factory=_int_factory)
    """User index 0."""

    user1: IntTensor = field(default_factory=_int_factory)
    """User index 1."""

    user2: IntTensor = field(default_factory=_int_factory)
    """User index 2."""

    user3: IntTensor = field(default_factory=_int_factory)
    """User index 3."""

    user4: IntTensor = field(default_factory=_int_factory)
    """User index 4."""

    user5: IntTensor = field(default_factory=_int_factory)
    """User index 5."""

    user6: IntTensor = field(default_factory=_int_factory)
    """User index 6."""

    user7: IntTensor = field(default_factory=_int_factory)
    """User index 7."""

    @property
    def shape(self) -> torch.Size:
        """Return shape of the KData object."""
        if not hasattr(self, '_memo'):
            self.check_invariants()
        return torch.Size(string_to_size('*#other 1 #k2 #k1 1', self._memo))


class UserValues(Dataclass):
    """User Values used in AcqInfo."""

    float1: FloatTensor = field(default_factory=_float_factory)
    float2: FloatTensor = field(default_factory=_float_factory)
    float3: FloatTensor = field(default_factory=_float_factory)
    float4: FloatTensor = field(default_factory=_float_factory)
    float5: FloatTensor = field(default_factory=_float_factory)
    float6: FloatTensor = field(default_factory=_float_factory)
    float7: FloatTensor = field(default_factory=_float_factory)
    float8: FloatTensor = field(default_factory=_float_factory)
    int1: IntTensor = field(default_factory=_int_factory)
    int2: IntTensor = field(default_factory=_int_factory)
    int3: IntTensor = field(default_factory=_int_factory)
    int4: IntTensor = field(default_factory=_int_factory)
    int5: IntTensor = field(default_factory=_int_factory)
    int6: IntTensor = field(default_factory=_int_factory)
    int7: IntTensor = field(default_factory=_int_factory)
    int8: IntTensor = field(default_factory=_int_factory)

    @property
    def shape(self) -> torch.Size:
        """Return shape of the KData object."""
        if not hasattr(self, '_memo'):
            self.check_invariants()
        return torch.Size(string_to_size('*#other 1 #k2 #k1 1', self._memo))


class PhysiologyTimestamps(Dataclass):
    """Time stamps relative to physiological triggering, e.g. ECG, in seconds."""

    timestamp1: FloatTensor = field(default_factory=_float_factory)
    timestamp2: FloatTensor = field(default_factory=_float_factory)
    timestamp3: FloatTensor = field(default_factory=_float_factory)

    @property
    def shape(self) -> torch.Size:
        """Return shape of the KData object."""
        if not hasattr(self, '_memo'):
            self.check_invariants()
        return torch.Size(string_to_size('*#other 1 #k2 #k1 1', self._memo))


class AcqInfo(Dataclass):
    """Acquisition information for each readout."""

    idx: Annotated[AcqIdx, ShapeAnnotation] = field(default_factory=AcqIdx)
    """Indices describing acquisitions (i.e. readouts)."""

    acquisition_time_stamp: Float64Tensor = field(default_factory=_float_factory)
    """Clock time stamp [s] (Siemens: seconds since midnight)"""

    flags: IntTensor = field(default_factory=_int_factory)
    """A bit mask of common attributes applicable to individual acquisition readouts."""

    orientation: Annotated[Rotation, ShapeAnnotation] = field(
        default_factory=lambda: Rotation.identity((1, 1, 1, 1, 1))
    )
    """Rotation describing the orientation of the readout, phase and slice encoding direction."""

    patient_table_position: SpatialDimensionTensor = field(default_factory=_position_factory)
    """Offset position of the patient table, in LPS coordinates [m]."""

    physiology_time_stamps: Annotated[PhysiologyTimestamps, ShapeAnnotation] = field(
        default_factory=PhysiologyTimestamps
    )
    """Time stamps relative to physiological triggering, e.g. ECG [s]."""

    position: SpatialDimensionTensor = field(default_factory=_position_factory)
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    sample_time_us: IntTensor = field(default_factory=_float_factory)
    """Readout bandwidth, as time between samples [us]."""

    user: Annotated[UserValues, ShapeAnnotation] = field(default_factory=UserValues)
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

    @property
    def shape(self) -> torch.Size:
        """Return shape of the KData object."""
        if not hasattr(self, '_memo'):
            self.check_invariants()
        return torch.Size(string_to_size('*#other 1 #k2 #k1 1', self._memo))
