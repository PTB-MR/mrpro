"""MR image data header (IHeader) dataclass."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from pydicom.dataset import Dataset
from pydicom.tag import Tag, TagType

from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues

MISC_TAGS = {'TimeAfterStart': 0x00191016}
UNKNOWN = 'unknown'


@dataclass(slots=True)
class IHeader(MoveDataMixin):
    """MR image data header."""

    b0: float | None
    """Magnetic field strength [T]."""

    fov: SpatialDimension[float]
    """Field of view [m]."""

    h1_freq: float | None
    """Lamor frequency of hydrogen nuclei [Hz]."""

    te: torch.Tensor | None
    """Echo time [s]."""

    ti: torch.Tensor | None
    """Inversion time [s]."""

    fa: torch.Tensor | None
    """Flip angle [rad]."""

    tr: torch.Tensor | None
    """Repetition time [s]."""

    patient_table_position: SpatialDimension[torch.Tensor] | None
    """Offset position of the patient table, in LPS coordinates [m]."""

    phase_dir: SpatialDimension[torch.Tensor] | None
    """Directional cosine of phase encoding (2D)."""

    position: SpatialDimension[torch.Tensor] | None
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    read_dir: SpatialDimension[torch.Tensor] | None
    """Directional cosine of readout/frequency encoding."""

    slice_dir: SpatialDimension[torch.Tensor] | None
    """Directional cosine of slice normal, i.e. cross-product of read_dir and phase_dir."""

    n_coils: int | None = None
    """Number of receiver coils."""

    datetime: datetime.datetime | None = None
    """Date and time of acquisition."""

    echo_train_length: int | None = 1
    """Number of echoes in a multi-echo acquisition."""

    seq_type: str = UNKNOWN
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
            b0=kheader.b0,
            fov=kheader.recon_fov,
            h1_freq=kheader.h1_freq,
            te=kheader.te,
            ti=kheader.ti,
            fa=kheader.fa,
            tr=kheader.tr,
            patient_table_position=kheader.acq_info.patient_table_position,
            phase_dir=kheader.acq_info.phase_dir,
            position=kheader.acq_info.position,
            read_dir=kheader.acq_info.read_dir,
            slice_dir=kheader.acq_info.slice_dir,
            n_coils=kheader.n_coils,
            datetime=kheader.datetime,
            echo_train_length=kheader.echo_train_length,
            seq_type=kheader.seq_type,
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

        def get_items_from_all_dicoms(name: TagType):
            """Get list of items for all dataset objects in the list."""
            return [get_item(ds, name) for ds in dicom_datasets]

        def get_float_items_from_all_dicoms(name: TagType):
            """Convert items to float."""
            items = get_items_from_all_dicoms(name)
            return [float(val) if val is not None else None for val in items]

        def make_unique_tensor(values: Sequence[float]) -> torch.Tensor | None:
            """If all the values are the same only return one."""
            if any(val is None for val in values):
                return None
            elif len(np.unique(values)) == 1:
                return torch.as_tensor([values[0]])
            else:
                return torch.as_tensor(values)

        # Conversion functions for units
        def ms_to_s(ms: torch.Tensor | None) -> torch.Tensor | None:
            return None if ms is None else ms / 1000

        def deg_to_rad(deg: torch.Tensor | None) -> torch.Tensor | None:
            return None if deg is None else torch.deg2rad(deg)

        fa = deg_to_rad(make_unique_tensor(get_float_items_from_all_dicoms('FlipAngle')))
        ti = ms_to_s(make_unique_tensor(get_float_items_from_all_dicoms('InversionTime')))
        tr = ms_to_s(make_unique_tensor(get_float_items_from_all_dicoms('RepetitionTime')))

        # get echo time(s). Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
        te_list = get_float_items_from_all_dicoms('EchoTime')
        if all(val is None for val in te_list):  # check if all entries are None
            te_list = get_float_items_from_all_dicoms('EffectiveEchoTime')
        te = ms_to_s(make_unique_tensor(te_list))

        fov_x_mm = get_float_items_from_all_dicoms('Rows')[0] * float(get_items_from_all_dicoms('PixelSpacing')[0][0])
        fov_y_mm = get_float_items_from_all_dicoms('Columns')[0] * float(
            get_items_from_all_dicoms('PixelSpacing')[0][1],
        )
        fov_z_mm = get_float_items_from_all_dicoms('SliceThickness')[0]
        fov = SpatialDimension(fov_x_mm / 1000.0, fov_y_mm / 1000.0, fov_z_mm / 1000.0)

        # Get misc parameters
        misc = {}
        for name in MISC_TAGS:
            misc[name] = make_unique_tensor(get_float_items_from_all_dicoms(MISC_TAGS[name]))
        return cls(fov=fov, te=te, ti=ti, fa=fa, tr=tr, misc=misc)

    def __repr__(self):
        """Representation method for IHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        out = f'FOV [m]: {self.fov!s}\n' f'TE [s]: {te}\nTI [s]: {ti}\nFlip angle [rad]: {fa}.'
        return out
