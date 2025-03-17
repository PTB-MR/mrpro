"""MR image data header (IHeader) dataclass."""

import dataclasses
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import cast

import numpy as np
import torch
from pydicom.dataset import Dataset
from pydicom.tag import Tag, TagType
from typing_extensions import Self, TypeVar

from mrpro.data.AcqInfo import AcqIdx, PhysiologyTimestamps
from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.reduce_repeat import reduce_repeat
from mrpro.utils.reshape import unsqueeze_right
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import deg_to_rad, mm_to_m, ms_to_s


def _int_factory() -> torch.Tensor:
    return torch.zeros(1, 1, 1, 1, 1, dtype=torch.int64)  # other, coil, z, y, x


def parse_datetime(datetime_str: str) -> datetime:
    """Parse datetime string with or without timezone information.

    Parameters
    ----------
    datetime_str
        Datetime string which may has timezone information.

    Returns
    -------
        Datetime
    """
    # Check if the string has a timezone (e.g., "+0000", "-0800", "Z")
    if re.search(r'([+-]\d{2}:?\d{2}|Z)$', datetime_str):
        dt = datetime.strptime(datetime_str, '%Y%m%d%H%M%S.%f%z')
        return dt.replace(tzinfo=None)  # Remove timezone info
    else:
        return datetime.strptime(datetime_str, '%Y%m%d%H%M%S.%f')


T = TypeVar('T')


def get_items(datasets: Sequence[Dataset], name: TagType, target_type: Callable[..., T]) -> list[T]:
    """Get a flattened list of converted items from each dataset for a given name or Tag.

    Parameters
    ----------
    name
        The name or tag to filter items.
    datasets
        The datasets to extract the items from.
    target_type
        The target type to convert the items into.

    Returns
    -------
        A list of converted items.
    """
    return [target_type(item.value) for ds in datasets for item in ds.iterall() if item.tag == Tag(name)]


def try_reduce_repeat(value: T) -> T:
    """Replace dimensions by singleton if possible, raise exception if spatial or coil dimension remains."""
    # TODO: Should be replaced by the CheckDataMixin and ReduceRepeatMixin
    match value:
        case torch.Tensor():
            tensor = reduce_repeat(value, 1e-6)
            if tensor.shape[-4:].numel() == 1:
                return cast(T, tensor)
        case Rotation():
            tensor = value.as_matrix()
            tensor = reduce_repeat(tensor, 1e-4, range(tensor.ndim - 1))
            if tensor.shape[-5:-2].numel() == 1:
                return cast(T, Rotation.from_matrix(tensor))
        case SpatialDimension():
            spatialdimension = value.apply(lambda x: reduce_repeat(x, 1e-6))
            if spatialdimension.shape[-4:].numel() == 1:
                return cast(T, spatialdimension)
        case list():
            return cast(T, list(value))
        case _:
            raise NotImplementedError('Unsupported Type')

    raise ValueError(f'Dimension mismatch. Spatial or coil dimension should be reduced to a single value. {value}')


@dataclass(slots=True)
class ImageIdx(MoveDataMixin):
    """Indices for each slice or volume.

    The different counters describe the use of each slice or volume in the image data.
    See MRD Image specification for more information.

    References
    ----------
    .. [1] MRD Image Specification
       https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html
    """

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

    @classmethod
    def from_acqidx(cls, idx: AcqIdx) -> Self:
        """Create ImageIdx object from AcqIdx object.

        This copies the indices except k1 and k2, which are not used in the image header.

        Parameters
        ----------
        idx
            Acquisition indices.
        """
        return cls(
            average=try_reduce_repeat(idx.average),
            slice=try_reduce_repeat(idx.slice),
            contrast=try_reduce_repeat(idx.contrast),
            phase=try_reduce_repeat(idx.phase),
            repetition=try_reduce_repeat(idx.repetition),
            set=try_reduce_repeat(idx.set),
            user0=try_reduce_repeat(idx.user0),
            user1=try_reduce_repeat(idx.user1),
            user2=try_reduce_repeat(idx.user2),
            user3=try_reduce_repeat(idx.user3),
            user4=try_reduce_repeat(idx.user4),
            user5=try_reduce_repeat(idx.user5),
            user6=try_reduce_repeat(idx.user6),
            user7=try_reduce_repeat(idx.user7),
        )


@dataclass(slots=True)
class IHeader(MoveDataMixin):
    """MR image data header."""

    resolution: SpatialDimension[float]
    """Pixel spacing [m/px]."""

    te: list[float] | torch.Tensor = field(default_factory=list)
    """Echo time [s]."""

    ti: list[float] | torch.Tensor = field(default_factory=list)
    """Inversion time [s]."""

    fa: list[float] | torch.Tensor = field(default_factory=list)
    """Flip angle [rad]."""

    tr: list[float] | torch.Tensor = field(default_factory=list)
    """Repetition time [s]."""

    _misc: dict = dataclasses.field(default_factory=dict)
    """Dictionary with miscellaneous parameters."""

    position: SpatialDimension[torch.Tensor] = field(
        default_factory=lambda: SpatialDimension(  # other, coil, z, y, x
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
        )
    )
    """Center of the image or volume [m]"""

    orientation: Rotation = field(default_factory=lambda: Rotation.identity((1, 1, 1, 1, 1)))
    """Orientation of the image or volume"""

    patient_table_position: SpatialDimension[torch.Tensor] = field(
        default_factory=lambda: SpatialDimension(  # other, coil, z, y, x
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
        )
    )
    """Offset position of the patient table [m]"""

    acquisition_time_stamp: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, 1, 1, 1))
    """Clock time stamp of the slice or volume [s]"""

    physiology_time_stamps: PhysiologyTimestamps = field(default_factory=PhysiologyTimestamps)
    """Time stamps relative to physiological triggering, e.g. ECG [s]."""

    idx: ImageIdx = field(default_factory=ImageIdx)
    """Image Counters. For each slice or volume, describe its use."""

    @classmethod
    def from_kheader(cls, header: KHeader) -> Self:
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        header
            MR raw data header (KHeader) containing required meta data.
        """
        resolution = header.recon_fov / header.recon_matrix

        # TODO: how to deal with different values for each acquisition?
        return cls(
            resolution=resolution,
            te=try_reduce_repeat(header.te),
            ti=try_reduce_repeat(header.ti),
            fa=try_reduce_repeat(header.fa),
            tr=try_reduce_repeat(header.tr),
            orientation=try_reduce_repeat(header.acq_info.orientation),
            position=try_reduce_repeat(header.acq_info.position.apply(lambda x: x)),
            patient_table_position=try_reduce_repeat(header.acq_info.patient_table_position.apply(lambda x: x)),
            acquisition_time_stamp=try_reduce_repeat(header.acq_info.acquisition_time_stamp.mean((-1, -2, -3), True)),
            physiology_time_stamps=PhysiologyTimestamps(
                try_reduce_repeat(header.acq_info.physiology_time_stamps.timestamp1.mean((-1, -2, -3), True)),
                try_reduce_repeat(header.acq_info.physiology_time_stamps.timestamp2.mean((-1, -2, -3), True)),
                try_reduce_repeat(header.acq_info.physiology_time_stamps.timestamp3.mean((-1, -2, -3), True)),
            ),
            idx=ImageIdx.from_acqidx(header.acq_info.idx),
        )

    @classmethod
    def from_dicom(cls, *dataset: Dataset) -> Self:
        """Read DICOM files and return IHeader object.

        Parameters
        ----------
        dataset
            one or multiple dataset objects containing the DICOM data.
        """
        # Ensure datasets are consistently single-frame or multi-frame 2D/3D
        number_of_frames = get_items(dataset, 'NumberOfFrames', int)
        if len(set(number_of_frames)) > 1:
            raise ValueError('Only DICOM files with the same number of frames can be stacked.')

        mr_acquisition_type = get_items(dataset, 'MRAcquisitionType', str)
        if len(set(mr_acquisition_type)) > 1:
            raise ValueError('Only DICOM files with the same MRAcquisitionType can be stacked.')

        pixel_spacing = get_items(dataset, 'PixelSpacing', lambda x: [float(val) for val in x])
        if not pixel_spacing or len(np.unique(torch.tensor(pixel_spacing), axis=0)) > 1:
            raise ValueError('Pixel spacing needs to be defined and the same for all.')

        slice_thickness = get_items(dataset, 'SliceThickness', float)
        if len(set(slice_thickness)) != 1:
            raise ValueError('Slice thickness needs to be defined and the same for all.')

        # The only mandatory field
        resolution = SpatialDimension(
            z=slice_thickness[0],
            y=pixel_spacing[0][1],
            x=pixel_spacing[0][0],
        ).apply_(mm_to_m)

        datasets_are_3d = (
            number_of_frames and number_of_frames[0] > 1 and mr_acquisition_type and mr_acquisition_type[0] == '3D'
        )
        n_volumes = number_of_frames[0] if number_of_frames and datasets_are_3d else 1

        header = cls(resolution=resolution)

        # For 3D datasets we currently want to save only one value per volume of fa, ti, tr, and te
        if fa := deg_to_rad(get_items(dataset, 'FlipAngle', float)[::n_volumes]):
            header.fa = fa
        if ti := ms_to_s(get_items(dataset, 'InversionTime', float)[::n_volumes]):
            header.ti = ti
        if tr := ms_to_s(get_items(dataset, 'RepetitionTime', float)[::n_volumes]):
            header.tr = tr
        te_ms = get_items(dataset, 'EchoTime', float)
        if not te_ms:  # Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
            te_ms = get_items(dataset, 'EffectiveEchoTime', float)
        if te_ms:
            header.te = ms_to_s(te_ms[::n_volumes])

        # Dicom orientation is described by two vectors in the format (x1, y1, z1, x2, y2, z2)
        if dcm_orientation := get_items(dataset, 'ImageOrientationPatient', lambda x: [float(val) for val in x]):
            basis_x = torch.tensor(dcm_orientation[0][:3])
            basis_y = torch.tensor(dcm_orientation[0][3:])
            # Calculate third basis vector by cross product
            basis_z = torch.cross(basis_x, basis_y, dim=-1)
            # Get orientation from basis vectors
            orientation = Rotation.from_directions(
                SpatialDimension.from_array_xyz(basis_x),
                SpatialDimension.from_array_xyz(basis_y),
                SpatialDimension.from_array_xyz(basis_z),
            )
        else:
            # Create default orientation for position shift
            orientation = Rotation.identity((1, 1, 1, 1, 1))
        header.orientation = orientation

        n_rows = get_items(dataset, 'Rows', int)[0]
        n_cols = get_items(dataset, 'Columns', int)[0]
        if n_rows and n_cols:
            shift = resolution * SpatialDimension(x=n_rows, y=n_cols, z=0) / 2
            # Get the shift vector for image position by applying the fov rotation
            shift = orientation(shift)
            # Get dicom position in [m] for x, y, z which is defined by the voxel center in the upper left corner
            dcm_position = SpatialDimension.from_array_xyz(
                get_items(dataset, 'ImagePositionPatient', lambda x: [float(val) for val in x])
            ).apply(mm_to_m)
            if dcm_position:
                # Shift dicom image position to the center by fov
                position = dcm_position + shift
                position = position.apply(lambda x: unsqueeze_right(x, 4))
                header.position = position

        # Calculate acquisition time stamps in s according to the reference time 0:00 am
        frame_time_dt = get_items(dataset, 'FrameReferenceDateTime', parse_datetime)
        if frame_time_dt:
            t0 = datetime.combine(frame_time_dt[0].date(), time(0, 0))
            header.acquisition_time_stamp = torch.tensor([(ft - t0).total_seconds() for ft in frame_time_dt])

        if dcm_cardiac_trigger := get_items(dataset, 'NominalCardiacTriggerDelayTime', float):
            header.physiology_time_stamps = PhysiologyTimestamps(timestamp1=ms_to_s(torch.tensor(dcm_cardiac_trigger)))

        # The in stack position accounts for the slice position for multi-file data with cardiac phases,
        # as well as for single file dicom with multiple slices. Index is reduced by 1 to start indexing at 0.
        # ToDO: Reshape indices?
        if phase_idx := get_items(dataset, 'TemporalPositionIndex', int):
            header.idx.phase = torch.tensor(phase_idx) - 1
        if slice_idx := get_items(dataset, 'InStackPositionNumber', int):
            header.idx.slice = torch.tensor(slice_idx) - 1

        return header

    def __repr__(self):
        """Representation method for IHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        tr = summarize_tensorvalues(self.tr)
        out = f'Resolution [m/pixel]: {self.resolution!s}\nTE [s]: {te}\nTR [s]: {tr}\nTI [s]: {ti}\n\
            Flip angle [rad]: {fa}.'
        return out
