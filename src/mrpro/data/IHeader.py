"""MR image data header (IHeader) dataclass."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar

import numpy as np
import torch
from pydicom.dataset import Dataset
from pydicom.tag import Tag, TagType

from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import mm_to_m, ms_to_s

MISC_TAGS = {'TimeAfterStart': 0x00191016}
UNKNOWN = 'unknown'


@dataclass(slots=True)
class IHeader(MoveDataMixin):
    """MR image data header."""

    fov: SpatialDimension[float]
    """Field of view [m]."""

    lamor_frequency_proton: float | None
    """Lamor frequency of hydrogen nuclei [Hz]."""

    te: torch.Tensor | None
    """Echo time [s]."""

    ti: torch.Tensor | None
    """Inversion time [s]."""

    fa: torch.Tensor | None
    """Flip angle [rad]."""

    tr: torch.Tensor | None
    """Repetition time [s]."""

    position: SpatialDimension[torch.Tensor] | None
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    orientation: Rotation | None
    """Rotation describing the orientation of the readout, phase and slice encoding direction."""

    patient_table_position: SpatialDimension[torch.Tensor] | None = None
    """Offset position of the patient table, in LPS coordinates [m]."""

    datetime: datetime.datetime | None = None
    """Date and time of acquisition."""

    echo_train_length: int | None = 1
    """Number of echoes in a multi-echo acquisition."""

    sequence_type: str = UNKNOWN
    """Type of sequence."""

    model: str = UNKNOWN
    """Scanner model."""

    vendor: str = UNKNOWN
    """Scanner vendor."""

    protocol_name: str = UNKNOWN
    """Name of the acquisition protocol."""

    measurement_id: str = UNKNOWN
    """Measurement ID."""

    patient_name: str = UNKNOWN
    """Name of the patient."""

    misc: dict = dataclasses.field(default_factory=dict)
    """Dictionary with miscellaneous parameters."""

    @classmethod
    def from_kheader(cls, kheader: KHeader) -> Self:
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """
        return cls(
            fov=kheader.recon_fov,
            lamor_frequency_proton=kheader.lamor_frequency_proton,
            te=kheader.te,
            ti=kheader.ti,
            fa=kheader.fa,
            tr=kheader.tr,
            patient_table_position=kheader.acq_info.patient_table_position,
            position=kheader.acq_info.position,
            orientation=kheader.acq_info.orientation,
            datetime=kheader.datetime,
            echo_train_length=kheader.echo_train_length,
            sequence_type=kheader.sequence_type,
            model=kheader.model,
            vendor=kheader.vendor,
            protocol_name=kheader.protocol_name,
            measurement_id=kheader.measurement_id,
            patient_name=kheader.patient_name,
        )

    @classmethod
    def from_dicom_list(cls, dicom_datasets: Sequence[Dataset]) -> Self:
        """Read DICOM files and return IHeader object.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """

        def get_item(dataset: Dataset, name: TagType):
            """Get item with a given name or Tag from a pydicom dataset."""
            # iterall is recursive, so it will find all items with the given name
            found_item = [item.value for item in dataset.iterall() if item.tag == Tag(name)]

            if len(found_item) == 0:
                return None
            elif len(found_item) == 1:
                return found_item[0]
            else:
                raise ValueError(f'Item {name} found {len(found_item)} times.')

        def get_items_as_list(name: TagType, item_function: Callable = lambda item: item):
            """Get list of items for all dataset objects in the list."""
            items = [get_item(ds, name) for ds in dicom_datasets]
            return [item_function(item) if item is not None else None for item in items]

        def make_unique_tensor(values: Sequence[float | int]) -> torch.Tensor | None:
            """If all the values are the same only return one."""
            if any(val is None for val in values):
                return None
            elif len(np.unique(values)) == 1:
                return torch.as_tensor([values[0]])
            else:
                return torch.as_tensor(values)

        T = TypeVar('T')

        def ensure_unique_entry(values: Sequence[T], dicom_tag: TagType) -> T:
            """Ensure there is only one unique entry."""
            if len(set(values)) == 1:
                return values[0]
            else:
                raise ValueError(f'Inconsistent entries for {dicom_tag} found.')

        def ensure_unique_string(values: Sequence[str], dicom_tag: TagType) -> str:
            """Ensure there is only one unique string."""
            if any(val is None for val in values):
                return UNKNOWN
            else:
                return ensure_unique_entry(values, dicom_tag)

        # Parameters which have to be the same for all entries in the dicom dataset
        lamor_frequency_proton = ensure_unique_entry(
            get_items_as_list('ImagingFrequency', lambda item: item), 'ImagingFrequency'
        )
        rows_x = ensure_unique_entry(get_items_as_list('Rows', lambda item: float(item)), 'Rows')
        columns_y = ensure_unique_entry(get_items_as_list('Columns', lambda item: float(item)), 'Columns')
        resolution_x = ensure_unique_entry(
            get_items_as_list('PixelSpacing', lambda item: mm_to_m(item[0])), 'PixelSpacing'
        )
        resolution_y = ensure_unique_entry(
            get_items_as_list('PixelSpacing', lambda item: mm_to_m(item[1])), 'PixelSpacing'
        )
        resolution_z = ensure_unique_entry(
            get_items_as_list('SliceThickness', lambda item: mm_to_m(item)), 'SliceThickness'
        )
        echo_train_length = ensure_unique_entry(
            get_items_as_list('EchoTrainLength', lambda item: int(item)), 'EchoTrainLength'
        )

        fov = SpatialDimension(resolution_z, resolution_y * columns_y, resolution_x * rows_x)

        sequence_type = ensure_unique_string(get_items_as_list('SequenceName'), 'SequenceName')
        model = ensure_unique_string(get_items_as_list(0x00081090), 'ManufacturersModelName')
        vendor = ensure_unique_string(get_items_as_list('Manufacturer'), 'Manufacturer')
        protocol_name = ensure_unique_string(get_items_as_list('ProtocolName'), 'ProtocolName')
        patient_name = ensure_unique_string(get_items_as_list('PatientName'), 'PatientName')

        # Parameters which can vary between the entries of the dicom dataset
        fa = make_unique_tensor(get_items_as_list('FlipAngle', lambda item: np.deg2rad(float(item))))
        ti = make_unique_tensor(get_items_as_list('InversionTime', lambda item: ms_to_s(float(item))))
        tr = make_unique_tensor(get_items_as_list('RepetitionTime', lambda item: ms_to_s(float(item))))

        # get echo time(s). Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
        te_list = get_items_as_list('EchoTime', lambda item: ms_to_s(float(item)))
        if all(val is None for val in te_list):  # check if all entries are None
            te_list = get_items_as_list('EffectiveEchoTime', lambda item: ms_to_s(float(item)))
        te = make_unique_tensor(te_list)

        # Image orientation and position has to be defined for all images or we set it to None
        image_orientation = get_items_as_list('ImageOrientationPatient', lambda item: torch.as_tensor(item))
        image_orientation_available = all(ori is not None for ori in image_orientation)
        read_dir = torch.stack(image_orientation)[:, :3] if image_orientation_available else None
        phase_dir = torch.stack(image_orientation)[:, 3:] if image_orientation_available else None
        slice_dir = torch.linalg.cross(read_dir, phase_dir) if read_dir is not None and phase_dir is not None else None
        orientation = (
            Rotation.from_matrix(torch.stack((slice_dir, phase_dir, read_dir), dim=-2))
            if read_dir is not None and phase_dir is not None and slice_dir is not None
            else None
        )

        # For dicom the image position is defined as the position of the top left voxel. In MRpro it is the centre of
        # the image. In order to calculate the MRpro position we need to to know the orientation
        def calc_position_from_dicom(
            position_xyz: list[float], fov: SpatialDimension, image_orientation: list[torch.Tensor]
        ):
            """Calculate the position as the centre of the image."""
            position_xyz_tensor = mm_to_m(torch.as_tensor(position_xyz))
            position_xyz_tensor += (
                fov.x / 2 * torch.stack(image_orientation)[:, :3] + fov.y / 2 * torch.stack(image_orientation)[:, 3:]
            )
            return position_xyz_tensor

        image_position = get_items_as_list('ImagePositionPatient')
        image_position_available = all(pos is not None for pos in image_position) and image_orientation_available
        position = (
            SpatialDimension.from_array_xyz(calc_position_from_dicom(image_position, fov, image_orientation))
            if image_position_available is not None
            else None
        )

        # Get the earliest date as the start of the entire data acquisition
        acq_date = get_items_as_list('AcquisitionDate')
        acq_time = get_items_as_list('AcquisitionTime')
        acq_date_time_available = all(date is not None for date in acq_date) and all(
            time is not None for time in acq_time
        )
        date_time_list = (
            [
                datetime.datetime.strptime(date + time, '%Y%m%d%H%M%S.%f')
                for date, time in zip(acq_date, acq_time, strict=True)
            ]
            if acq_date_time_available
            else None
        )
        date_time = sorted(date_time_list)[0] if date_time_list is not None else None

        # Get misc parameters
        misc = {}
        for name in MISC_TAGS:
            misc[name] = make_unique_tensor(get_items_as_list(MISC_TAGS[name], lambda item: float(item)))
        return cls(
            fov=fov,
            lamor_frequency_proton=lamor_frequency_proton,
            te=te,
            ti=ti,
            fa=fa,
            tr=tr,
            position=position,
            orientation=orientation,
            datetime=date_time,
            echo_train_length=echo_train_length,
            sequence_type=sequence_type,
            model=model,
            vendor=vendor,
            protocol_name=protocol_name,
            patient_name=patient_name,
            misc=misc,
        )

    def __repr__(self):
        """Representation method for IHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        out = f'FOV [m]: {self.fov!s}\n' f'TE [s]: {te}\nTI [s]: {ti}\nFlip angle [rad]: {fa}.'
        return out
