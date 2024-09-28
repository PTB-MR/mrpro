"""Acquisition information dataclass."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar

import ismrmrd
import numpy as np
import torch

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension

# Conversion functions for units
T = TypeVar('T', float, torch.Tensor)


def ms_to_s(ms: T) -> T:
    """Convert ms to s."""
    return ms / 1000


def mm_to_m(m: T) -> T:
    """Convert mm to m."""
    return m / 1000


def s_to_ms(ms: T) -> T:
    """Convert s to ms."""
    return ms * 1000


def m_to_mm(m: T) -> T:
    """Convert m to mm."""
    return m * 1000


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
class AcqInfo(MoveDataMixin):
    """Acquisition information for each readout."""

    idx: AcqIdx
    """Indices describing acquisitions (i.e. readouts)."""

    acquisition_time_stamp: torch.Tensor
    """Clock time stamp. Not in s but in vendor-specific time units (e.g. 2.5ms for Siemens)"""

    active_channels: torch.Tensor
    """Number of active receiver coil elements."""

    available_channels: torch.Tensor
    """Number of available receiver coil elements."""

    center_sample: torch.Tensor
    """Index of the readout sample corresponding to k-space center (zero indexed)."""

    channel_mask: torch.Tensor
    """Bit mask indicating active coils (64*16 = 1024 bits)."""

    discard_post: torch.Tensor
    """Number of readout samples to be discarded at the end (e.g. if the ADC is active during gradient events)."""

    discard_pre: torch.Tensor
    """Number of readout samples to be discarded at the beginning (e.g. if the ADC is active during gradient events)"""

    encoding_space_ref: torch.Tensor
    """Indexed reference to the encoding spaces enumerated in the MRD (xml) header."""

    flags: torch.Tensor
    """A bit mask of common attributes applicable to individual acquisition readouts."""

    measurement_uid: torch.Tensor
    """Unique ID corresponding to the readout."""

    number_of_samples: torch.Tensor
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

    scan_counter: torch.Tensor
    """Zero-indexed incrementing counter for readouts."""

    slice_dir: SpatialDimension[torch.Tensor]
    """Directional cosine of slice normal, i.e. cross-product of read_dir and phase_dir."""

    trajectory_dimensions: torch.Tensor  # =3. We only support 3D Trajectories: kz always exists.
    """Dimensionality of the k-space trajectory vector."""

    user_float: torch.Tensor
    """User-defined float parameters."""

    user_int: torch.Tensor
    """User-defined int parameters."""

    version: torch.Tensor
    """Major version number."""

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
            # we have to convert first as pytoch cant create tensors from np.uint16 arrays
            # we use int32 for uint16 and int64 for uint32 to fit largest values.
            match data.dtype:
                case np.uint16:
                    data = data.astype(np.int32)
                case np.uint32 | np.uint64:
                    data = data.astype(np.int64)
            # Remove any uncessary dimensions
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

        def spatialdimension_2d(
            data: np.ndarray, conversion: Callable[[torch.Tensor], torch.Tensor] | None = None
        ) -> SpatialDimension[torch.Tensor]:
            # Ensure spatial dimension is (k1*k2*other, 1, 3)
            if data.ndim != 2:
                raise ValueError('Spatial dimension is expected to be of shape (N,3)')
            data = data[:, None, :]
            # all spatial dimensions are float32
            return SpatialDimension[torch.Tensor].from_array_xyz(torch.tensor(data.astype(np.float32)), conversion)

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

        acq_info = cls(
            idx=acq_idx,
            acquisition_time_stamp=tensor_2d(headers['acquisition_time_stamp']),
            active_channels=tensor_2d(headers['active_channels']),
            available_channels=tensor_2d(headers['available_channels']),
            center_sample=tensor_2d(headers['center_sample']),
            channel_mask=tensor_2d(headers['channel_mask']),
            discard_post=tensor_2d(headers['discard_post']),
            discard_pre=tensor_2d(headers['discard_pre']),
            encoding_space_ref=tensor_2d(headers['encoding_space_ref']),
            flags=tensor_2d(headers['flags']),
            measurement_uid=tensor_2d(headers['measurement_uid']),
            number_of_samples=tensor_2d(headers['number_of_samples']),
            patient_table_position=spatialdimension_2d(headers['patient_table_position'], mm_to_m),
            phase_dir=spatialdimension_2d(headers['phase_dir']),
            physiology_time_stamp=tensor_2d(headers['physiology_time_stamp']),
            position=spatialdimension_2d(headers['position'], mm_to_m),
            read_dir=spatialdimension_2d(headers['read_dir']),
            sample_time_us=tensor_2d(headers['sample_time_us']),
            scan_counter=tensor_2d(headers['scan_counter']),
            slice_dir=spatialdimension_2d(headers['slice_dir']),
            trajectory_dimensions=tensor_2d(headers['trajectory_dimensions']).fill_(3),  # see above
            user_float=tensor_2d(headers['user_float']),
            user_int=tensor_2d(headers['user_int']),
            version=tensor_2d(headers['version']),
        )
        return acq_info

    def add_ismrmrd_acquisition_info(
        self, acquisition: ismrmrd.Acquisition, other: int, k2: int, k1: int
    ) -> ismrmrd.Acquisition:
        """ISMRMRD acquisition information for single acquisition."""
        acquisition.idx.kspace_encode_step_1 = self.idx.k1[other, k2, k1]
        acquisition.idx.kspace_encode_step_2 = self.idx.k2[other, k2, k1]
        acquisition.idx.average = self.idx.average[other, k2, k1]
        acquisition.idx.slice = self.idx.slice[other, k2, k1]
        acquisition.idx.contrast = self.idx.contrast[other, k2, k1]
        acquisition.idx.phase = self.idx.phase[other, k2, k1]
        acquisition.idx.repetition = self.idx.repetition[other, k2, k1]
        acquisition.idx.set = self.idx.set[other, k2, k1]
        acquisition.idx.segment = self.idx.segment[other, k2, k1]
        acquisition.idx.user = (
            self.idx.user0[other, k2, k1],
            self.idx.user1[other, k2, k1],
            self.idx.user2[other, k2, k1],
            self.idx.user3[other, k2, k1],
            self.idx.user4[other, k2, k1],
            self.idx.user5[other, k2, k1],
            self.idx.user6[other, k2, k1],
            self.idx.user7[other, k2, k1],
        )

        # active_channesl, number_of_samples and trajectory_dimensions are read-only and cannot be set
        acquisition.acquisition_time_stamp = self.acquisition_time_stamp[other, k2, k1, 0]
        acquisition.available_channels = self.available_channels[other, k2, k1, 0]
        acquisition.center_sample = self.center_sample[other, k2, k1, 0]
        acquisition.channel_mask = tuple(self.channel_mask[other, k2, k1, :])
        acquisition.discard_post = self.discard_post[other, k2, k1, 0]
        acquisition.discard_pre = self.discard_pre[other, k2, k1, 0]
        acquisition.encoding_space_ref = self.encoding_space_ref[other, k2, k1, 0]
        acquisition.measurement_uid = self.measurement_uid[other, k2, k1, 0]
        acquisition.patient_table_position = tuple(
            [m_to_mm(getattr(self.patient_table_position, label)[other, k2, k1, 0]) for label in ['x', 'y', 'z']]
        )
        acquisition.phase_dir = tuple([getattr(self.phase_dir, label)[other, k2, k1, 0] for label in ['x', 'y', 'z']])
        acquisition.physiology_time_stamp = tuple(self.physiology_time_stamp[other, k2, k1, :])
        acquisition.position = tuple(
            [m_to_mm(getattr(self.position, label)[other, k2, k1, 0]) for label in ['x', 'y', 'z']]
        )
        acquisition.read_dir = tuple([getattr(self.read_dir, label)[other, k2, k1, 0] for label in ['x', 'y', 'z']])
        acquisition.sample_time_us = self.sample_time_us[other, k2, k1, 0]
        acquisition.scan_counter = self.scan_counter[other, k2, k1, 0]
        acquisition.slice_dir = tuple([getattr(self.slice_dir, label)[other, k2, k1, 0] for label in ['x', 'y', 'z']])
        acquisition.user_float = tuple(self.user_float[other, k2, k1, :])
        acquisition.user_int = tuple(self.user_int[other, k2, k1, :])
        acquisition.version = self.version[other, k2, k1, 0]
        return acquisition
