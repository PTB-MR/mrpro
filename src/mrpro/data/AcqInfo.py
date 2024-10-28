"""Acquisition information dataclass."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import einops
import ismrmrd
import numpy as np
import torch
from typing_extensions import Self, TypeVar

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension

# Conversion functions for units
T = TypeVar('T', float, torch.Tensor)

# We use this for runtime dtype checking in the dataclasses
LongTensor: TypeAlias = torch.Tensor | torch.LongTensor | torch.IntTensor


def ms_to_s(ms: T) -> T:
    """Convert ms to s."""
    return ms / 1000


def mm_to_m(m: T) -> T:
    """Convert mm to m."""
    return m / 1000


class _InvariantsAcqInfo:
    __slots__ = '__broadcasted_shape', '__typehints'
    __broadcasted_shape: torch.Size | None

    def __post_init__(self):
        self._check_invariants()

    def _check_invariants(self):
        shapes = []
        for name in self.__slots__:
            expected_type = self.__annotations__[name]
            value = getattr(self, name)
            if not isinstance(value, expected_type):
                raise TypeError(f'{name} must be of type {expected_type}, got {type(value)} instead')
            if hasattr(value, 'shape'):
                # isinstance(value, torch.Tensor | SpatialDimension | Rotation):
                shape = value.shape
                if len(shape) < 5:
                    raise ValueError(f'{name} must have at least 5 dimensions')
                if shape[-1] != 1:
                    raise ValueError(f'{name} must have a k0 dimension of size 1')
                if shape[-4] != 1:
                    raise ValueError(f'{name} must have a coil dimension of size 1')
                shapes.append(value.shape)

            if hasattr(value, 'dtype') and value.dtype.is_complex:
                raise ValueError(f'{name} must not be complex.')
            if expected_type == LongTensor and value.dtype not in (
                torch.int64,
                torch.uint64,
                torch.int32,
                torch.uint32,
            ):
                raise ValueError(f'{name} must be integer.')

            elif hasattr(value, 'broadcasted_shape'):
                shapes.append(value.broadcasted_shape)
        try:
            broadcasted_shape = torch.broadcast_shapes(*shapes)
        except RuntimeError:
            raise ValueError(f'The Acquisition information tensors {self.__slots__} must be broadcastable.') from None

        self.__broadcasted_shape = broadcasted_shape

    @property
    def broadcasted_shape(self) -> torch.Size:
        assert self.__broadcasted_shape is not None  # noqa: S101 # mypy hint
        return self.__broadcasted_shape


@dataclass(slots=True)
class AcqIdx(MoveDataMixin, _InvariantsAcqInfo):
    """Acquisition index for each readout."""

    k1: LongTensor
    """First phase encoding."""

    k2: LongTensor
    """Second phase encoding."""

    average: LongTensor
    """Signal average."""

    slice: LongTensor
    """Slice number (multi-slice 2D)."""

    contrast: LongTensor
    """Echo number in multi-echo."""

    phase: LongTensor
    """Cardiac phase."""

    repetition: LongTensor
    """Counter in repeated/dynamic acquisitions."""

    set: LongTensor
    """Sets of different preparation, e.g. flow encoding, diffusion weighting."""

    segment: LongTensor
    """Counter for segmented acquisitions."""

    user0: LongTensor
    """User index 0."""

    user1: LongTensor
    """User index 1."""

    user2: LongTensor
    """User index 2."""

    user3: LongTensor
    """User index 3."""

    user4: LongTensor
    """User index 4."""

    user5: LongTensor
    """User index 5."""

    user6: LongTensor
    """User index 6."""

    user7: LongTensor
    """User index 7."""


@dataclass(slots=True)
class UserValues(MoveDataMixin, _InvariantsAcqInfo):
    """User-defined values for each readout."""

    float0: torch.Tensor
    """User float 0."""

    float1: torch.Tensor
    """User float 1."""

    float2: torch.Tensor
    """User float 2."""

    float3: torch.Tensor
    """User float 3."""

    float4: torch.Tensor
    """User float 4."""

    float5: torch.Tensor
    """User float 5."""

    float6: torch.Tensor
    """User float 6."""

    float7: torch.Tensor
    """User float 7."""

    int0: LongTensor
    """User int 0."""

    int1: LongTensor
    """User int 1."""

    int2: LongTensor
    """User int 2."""

    int3: LongTensor
    """User int 3."""

    int4: LongTensor
    """User int 4."""

    int5: LongTensor
    """User int 5."""

    int6: LongTensor
    """User int 6."""

    int7: LongTensor
    """User int 7."""


@dataclass(slots=True)
class AcqInfo(MoveDataMixin):
    """Acquisition information for each readout."""

    idx: AcqIdx
    """Indices describing acquisitions (i.e. readouts)."""

    acquisition_time_stamp: torch.Tensor
    """Clock time stamp. Not in s but in vendor-specific time units (e.g. 2.5ms for Siemens)"""

    active_channels: LongTensor
    """Number of active receiver coil elements."""

    available_channels: LongTensor
    """Number of available receiver coil elements."""

    center_sample: LongTensor
    """Index of the readout sample corresponding to k-space center (zero indexed)."""

    channel_mask: LongTensor
    """Bit mask indicating active coils (64*16 = 1024 bits)."""

    discard_post: LongTensor
    """Number of readout samples to be discarded at the end (e.g. if the ADC is active during gradient events)."""

    discard_pre: LongTensor
    """Number of readout samples to be discarded at the beginning (e.g. if the ADC is active during gradient events)"""

    encoding_space_ref: LongTensor
    """Indexed reference to the encoding spaces enumerated in the MRD (xml) header."""

    flags: LongTensor
    """A bit mask of common attributes applicable to individual acquisition readouts."""

    measurement_uid: LongTensor
    """Unique ID corresponding to the readout."""

    number_of_samples: LongTensor
    """Number of sample points per readout (readouts may have different number of sample points)."""

    patient_table_position: SpatialDimension[torch.Tensor]
    """Offset position of the patient table, in LPS coordinates [m]."""

    phase_dir: SpatialDimension[torch.Tensor]
    """Directional cosine of phase encoding (2D)."""

    physiology_time_stamp: torch.Tensor
    """Time stamps relative to physiological triggering, e.g. ECG. Not in s but in vendor-specific time units"""

    position: SpatialDimension[torch.Tensor]
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    read_dir: SpatialDimension[torch.Tensor]
    """Directional cosine of readout/frequency encoding."""

    sample_time_us: torch.Tensor
    """Readout bandwidth, as time between samples [us]."""

    scan_counter: LongTensor
    """Zero-indexed incrementing counter for readouts."""

    slice_dir: SpatialDimension[torch.Tensor]
    """Directional cosine of slice normal, i.e. cross-product of read_dir and phase_dir."""

    trajectory_dimensions: LongTensor  # =3. We only support 3D Trajectories: kz always exists.
    """Dimensionality of the k-space trajectory vector."""

    user: UserValues
    """User-defined values for each readout."""

    version: LongTensor
    """Major version number"""

    @classmethod
    def from_ismrmrd_acquisitions(cls, acquisitions: Sequence[ismrmrd.Acquisition]) -> Self:
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
            np.array([memoryview(a._head).cast('B') for a in acquisitions]),
            dtype=acquisition_head_dtype,
        )

        idx = headers['idx']

        def tensor(data: np.ndarray) -> torch.Tensor:
            """Convert to tensor with shape (other=1, coil=1, k2=1, k1=n, k0=1)."""
            # we have to convert first as pytoch cant create tensors from np.uint16 arrays
            # we use int32 for uint16 and int64 for uint32 to fit largest values.
            match data.dtype:
                case np.uint16:
                    data = data.astype(np.int32)
                case np.uint32 | np.uint64:
                    data = data.astype(np.int64)
            tensor = torch.from_numpy(data)
            tensor = einops.repeat(tensor, '... -> other coil k2 (...) k0', other=1, coil=1, k2=1, k0=1)
            return tensor

            # # Ensure that data is (k1*k2*other, >=1)
            # if data_tensor.ndim == 1:
            #     data_tensor = data_tensor[:, None]
            # elif data_tensor.ndim == 0:
            #     data_tensor = data_tensor[None, None]
            # return data_tensor

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
        user_values = UserValues(
            float0=tensor(headers['user_float'][:, 0]),
            float1=tensor(headers['user_float'][:, 1]),
            float2=tensor(headers['user_float'][:, 2]),
            float3=tensor(headers['user_float'][:, 3]),
            float4=tensor(headers['user_float'][:, 4]),
            float5=tensor(headers['user_float'][:, 5]),
            float6=tensor(headers['user_float'][:, 6]),
            float7=tensor(headers['user_float'][:, 7]),
            int0=tensor(headers['user_int'][:, 0]),
            int1=tensor(headers['user_int'][:, 1]),
            int2=tensor(headers['user_int'][:, 2]),
            int3=tensor(headers['user_int'][:, 3]),
            int4=tensor(headers['user_int'][:, 4]),
            int5=tensor(headers['user_int'][:, 5]),
            int6=tensor(headers['user_int'][:, 6]),
            int7=tensor(headers['user_int'][:, 7]),
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
            patient_table_position=spatialdimension_2d(headers['patient_table_position'], mm_to_m),
            phase_dir=spatialdimension_2d(headers['phase_dir']),
            physiology_time_stamp=tensor(headers['physiology_time_stamp']),
            position=spatialdimension_2d(headers['position'], mm_to_m),
            read_dir=spatialdimension_2d(headers['read_dir']),
            sample_time_us=tensor(headers['sample_time_us']),
            scan_counter=tensor(headers['scan_counter']),
            slice_dir=spatialdimension_2d(headers['slice_dir']),
            trajectory_dimensions=tensor(headers['trajectory_dimensions']).fill_(3),  # see above
            user=user_values,
            version=tensor(headers['version']),
        )
        return acq_info
