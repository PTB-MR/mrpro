"""Base dataclass for data header."""

from __future__ import annotations

import datetime
from abc import ABC
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
from mrpro.utils.unit_conversion import mm_to_m

MISC_TAGS = {'TimeAfterStart': 0x00191016}
UNKNOWN = 'unknown'
T = TypeVar('T')


@dataclass(slots=True)
class Header(MoveDataMixin, ABC):
    """MR data header."""

    fov: SpatialDimension[float]
    """Field of view [m]."""

    lamor_frequency_proton: float | None
    """Lamor frequency of hydrogen nuclei [Hz]."""

    position: SpatialDimension[torch.Tensor] | None
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    orientation: Rotation | None
    """Rotation describing the orientation of the readout, phase and slice encoding direction."""

    patient_table_position: SpatialDimension[torch.Tensor] | None
    """Offset position of the patient table, in LPS coordinates [m]."""

    datetime: datetime.datetime | None
    """Date and time of acquisition."""

    sequence_type: str
    """Type of sequence."""

    model: str
    """Scanner model."""

    vendor: str
    """Scanner vendor."""

    protocol_name: str
    """Name of the acquisition protocol."""

    measurement_id: str
    """Measurement ID."""

    patient_name: str
    """Name of the patient."""

    misc: dict
    """Dictionary with miscellaneous parameters."""

    @classmethod
    def from_kheader(cls, kheader: KHeader) -> Self:
        """Create Header object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """
        return cls(
            fov=kheader.recon_fov,
            lamor_frequency_proton=kheader.lamor_frequency_proton,
            patient_table_position=kheader.acq_info.patient_table_position,
            position=kheader.acq_info.position,
            orientation=kheader.acq_info.orientation,
            datetime=kheader.datetime,
            sequence_type=kheader.sequence_type,
            model=kheader.model,
            vendor=kheader.vendor,
            protocol_name=kheader.protocol_name,
            measurement_id=kheader.measurement_id,
            patient_name=kheader.patient_name,
            misc={},
        )

    @staticmethod
    def get_item(dataset: Dataset, name: TagType) -> None | float | int | str:
        """Get item with a given name or Tag from a pydicom dataset."""
        # iterall is recursive, so it will find all items with the given name
        found_item = [item.value for item in dataset.iterall() if item.tag == Tag(name)]

        if len(found_item) == 0:
            return None
        elif len(found_item) == 1:
            return found_item[0]
        else:
            raise ValueError(f'Item {name} found {len(found_item)} times.')

    @staticmethod
    def get_items_as_list(
        datasets: Sequence[Dataset], name: TagType, item_function: Callable = lambda item: item
    ) -> Sequence:
        """Get list of items for all dataset objects in the list."""
        items = [Header.get_item(ds, name) for ds in datasets]
        return [item_function(item) if item is not None else None for item in items]

    @staticmethod
    def make_unique_tensor(values: Sequence[float | int]) -> torch.Tensor | None:
        """If all the values are the same only return one."""
        if any(val is None for val in values):
            return None
        elif len(np.unique(values)) == 1:
            return torch.as_tensor([values[0]])
        else:
            return torch.as_tensor(values)

    @staticmethod
    def ensure_unique_entry(values: Sequence[T], dicom_tag: TagType) -> T:
        """Ensure there is only one unique entry."""
        if len(set(values)) == 1:
            return values[0]
        else:
            raise ValueError(f'Inconsistent entries for {dicom_tag} found.')

    @staticmethod
    def ensure_unique_string(values: Sequence[str], dicom_tag: TagType) -> str:
        """Ensure there is only one unique string."""
        if any(val is None for val in values):
            return UNKNOWN
        else:
            return Header.ensure_unique_entry(values, dicom_tag)

    @staticmethod
    def attributes_from_dicom_list(dicom_datasets: Sequence[Dataset]) -> dict:
        """Read DICOM files and dictionary with header attributes.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """
        # Parameters which have to be the same for all entries in the dicom dataset
        lamor_frequency_proton = Header.ensure_unique_entry(
            Header.get_items_as_list(dicom_datasets, 'ImagingFrequency', lambda item: item), 'ImagingFrequency'
        )
        rows_x = Header.ensure_unique_entry(
            Header.get_items_as_list(dicom_datasets, 'Rows', lambda item: float(item)), 'Rows'
        )
        columns_y = Header.ensure_unique_entry(
            Header.get_items_as_list(dicom_datasets, 'Columns', lambda item: float(item)), 'Columns'
        )
        resolution_x = Header.ensure_unique_entry(
            Header.get_items_as_list(dicom_datasets, 'PixelSpacing', lambda item: mm_to_m(item[0])), 'PixelSpacing'
        )
        resolution_y = Header.ensure_unique_entry(
            Header.get_items_as_list(dicom_datasets, 'PixelSpacing', lambda item: mm_to_m(item[1])), 'PixelSpacing'
        )
        resolution_z = Header.ensure_unique_entry(
            Header.get_items_as_list(dicom_datasets, 'SliceThickness', lambda item: mm_to_m(item)), 'SliceThickness'
        )

        fov = SpatialDimension(resolution_z, resolution_y * columns_y, resolution_x * rows_x)

        sequence_type = Header.ensure_unique_string(
            Header.get_items_as_list(dicom_datasets, 'SequenceName'), 'SequenceName'
        )
        model = Header.ensure_unique_string(
            Header.get_items_as_list(dicom_datasets, 0x00081090), 'ManufacturersModelName'
        )
        vendor = Header.ensure_unique_string(Header.get_items_as_list(dicom_datasets, 'Manufacturer'), 'Manufacturer')
        protocol_name = Header.ensure_unique_string(
            Header.get_items_as_list(dicom_datasets, 'ProtocolName'), 'ProtocolName'
        )
        patient_name = Header.ensure_unique_string(
            Header.get_items_as_list(dicom_datasets, 'PatientName'), 'PatientName'
        )

        # Image orientation and position has to be defined for all images or we set it to None
        def dicom_orientation_to_header_orientation(image_orientation: Sequence) -> Rotation | None:
            if any(ori is not None for ori in image_orientation):
                return None
            else:
                orientation = [torch.as_tensor(item) for item in image_orientation]
                read_dir = torch.stack(orientation)[:, :3]
                phase_dir = torch.stack(orientation)[:, 3:]
                slice_dir = torch.linalg.cross(read_dir, phase_dir)
                return Rotation.from_matrix(torch.stack((slice_dir, phase_dir, read_dir), dim=-2))

        image_orientation = Header.get_items_as_list(dicom_datasets, 'ImageOrientationPatient')
        orientation = dicom_orientation_to_header_orientation(image_orientation)

        # For dicom the image position is defined as the position of the top left voxel. In MRpro it is the centre of
        # the image. In order to calculate the MRpro position we need to to know the orientation
        def dicom_position_to_header_position(
            image_position: Sequence, image_orientation: Sequence, fov: SpatialDimension
        ) -> SpatialDimension | None:
            if any(ori is not None for ori in image_orientation) or any(pos is not None for pos in image_position):
                return None
            else:
                orientation = [torch.as_tensor(item) for item in image_orientation]
                position = mm_to_m(torch.as_tensor(image_position))
                position += fov.x / 2 * torch.stack(orientation)[:, :3] + fov.y / 2 * torch.stack(orientation)[:, 3:]
                return SpatialDimension.from_array_xyz(position)

        image_position = Header.get_items_as_list(dicom_datasets, 'ImagePositionPatient')
        position = dicom_position_to_header_position(image_position, image_orientation, fov)

        # Get the earliest date as the start of the entire data acquisition
        acq_date = Header.get_items_as_list(dicom_datasets, 'AcquisitionDate')
        acq_time = Header.get_items_as_list(dicom_datasets, 'AcquisitionTime')
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
        return {
            'fov': fov,
            'lamor_frequency_proton': lamor_frequency_proton,
            'position': position,
            'orientation': orientation,
            'datetime': date_time,
            'sequence_type': sequence_type,
            'model': model,
            'vendor': vendor,
            'protocol_name': protocol_name,
            'patient_name': patient_name,
        }

    @classmethod
    def from_dicom_list(cls, dicom_datasets: Sequence[Dataset]) -> Self:
        """Read DICOM files and return Header object.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """
        return cls(**cls.attributes_from_dicom_list(dicom_datasets), misc={})
