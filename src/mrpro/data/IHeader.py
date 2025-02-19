"""MR image data header (IHeader) dataclass."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from einops import repeat
from pydicom import multival, valuerep
from pydicom.dataset import Dataset
from pydicom.tag import Tag, TagType
from typing_extensions import Self

from mrpro.data.AcqInfo import PhysiologyTimestamps
from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.remove_repeat import remove_repeat
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import deg_to_rad, mm_to_m, ms_to_s

MISC_TAGS = {'TimeAfterStart': 0x00191016}


def _int_factory() -> torch.Tensor:
    return torch.zeros(1, 1, dtype=torch.int64)


@dataclass(slots=True)
class ImageIdx(MoveDataMixin):
    """Acquisition index for each readout."""

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


@dataclass(slots=True)
class IHeader(MoveDataMixin):
    """MR image data header."""

    resolution: SpatialDimension[float]
    """Pixel spacing [m/px]."""

    te: torch.Tensor | None = None
    """Echo time [s]."""

    ti: torch.Tensor | None = None
    """Inversion time [s]."""

    fa: torch.Tensor | None = None
    """Flip angle [rad]."""

    tr: torch.Tensor | None = None
    """Repetition time [s]."""

    _misc: dict = dataclasses.field(default_factory=dict)
    """Dictionary with miscellaneous parameters."""

    position: SpatialDimension[torch.Tensor] = field(
        default_factory=lambda: SpatialDimension(
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
        )
    )
    """Center of the excited volume"""

    orientation: Rotation = field(default_factory=lambda: Rotation.identity((1, 1, 1, 1, 1)))
    """Orientation of the image"""

    patient_table_position: SpatialDimension[torch.Tensor] = field(
        default_factory=lambda: SpatialDimension(
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
        )
    )
    """Offset position of the patient table"""

    acquisition_time_stamp: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, 1, 1, 1))

    physiology_time_stamps: PhysiologyTimestamps = field(default_factory=PhysiologyTimestamps)

    idx: ImageIdx = field(default_factory=ImageIdx)

    @classmethod
    def from_kheader(cls, kheader: KHeader, resolution: SpatialDimension) -> Self:
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        resolution
            Pixel Spacing
        """
        return cls(resolution=resolution, te=kheader.te, ti=kheader.ti, fa=kheader.fa, tr=kheader.tr)

    @classmethod
    def from_dicoms(cls, dicom_datasets: Sequence[Dataset]) -> Self:
        """Read DICOM files and return IHeader object.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """

        def convert_pydicom_type(item) -> float | int | list[float | int | list | None] | None:  # noqa: ANN001
            """Convert item to type given by pydicom."""
            if isinstance(item, valuerep.DSfloat | valuerep.DSdecimal | valuerep.ISfloat):
                return float(item)
            elif isinstance(item, valuerep.IS):
                return int(item)
            elif isinstance(item, multival.MultiValue):
                return [convert_pydicom_type(it) for it in item]
            else:
                return None

        def get_items(name: TagType) -> list:
            """Get a flattened list of converted items from each dataset for a given name or Tag."""
            return [
                item
                for ds in dicom_datasets
                for item in ([convert_pydicom_type(it.value) for it in ds.iterall() if it.tag == Tag(name)] or [None])
            ]

        def make_unique_tensor(values: Sequence[float]) -> torch.Tensor | None:
            """If all the values are the same only return one."""
            if any(val is None for val in values):
                return None
            elif len(np.unique(values)) == 1:
                return torch.as_tensor([values[0]])
            else:
                return torch.as_tensor(values)

        def as_5d_tensor(values: Sequence[float]) -> torch.Tensor:
            """Convert a list of values to a 5d tensor."""
            tensor = torch.as_tensor(values)
            tensor = repeat(tensor, 'values-> values 1 1 1 1')
            tensor = remove_repeat(tensor, 1e-12)
            return tensor

        # Ensure datasets are consistently single-frame or multi-frame 2D/3D
        number_of_frames = get_items('NumberOfFrames')
        if len(set(number_of_frames)) > 1:
            raise ValueError('Only DICOM files with the same number of frames can be stacked.')

        mr_acquisition_type = get_items('MRAcquisitionType')
        if len(set(mr_acquisition_type)) > 1:
            raise ValueError('Only DICOM files with the same MRAcquisitionType can be stacked.')

        datasets_are_3d = number_of_frames[0] is not None and number_of_frames[0] > 1 and mr_acquisition_type[0] == '3D'
        n_volumes = number_of_frames[0] if datasets_are_3d else 1

        pixel_spacing = get_items('PixelSpacing')
        if pixel_spacing[0] is None or len(np.unique(torch.tensor(pixel_spacing), axis=0)) > 1:
            raise ValueError('Pixel spacing needs to be defined and the same for all.')

        slice_thickness = get_items('SliceThickness')
        if len(set(slice_thickness)) > 1 or slice_thickness[0] is None:
            raise ValueError('Slice thickness needs to be defined and the same for all.')

        resolution = SpatialDimension(
            z=slice_thickness[0],
            y=pixel_spacing[0][1],
            x=pixel_spacing[0][0],
        ).apply_(mm_to_m)

        fa_deg = get_items('FlipAngle')[::n_volumes]
        ti_ms = get_items('InversionTime')[::n_volumes]
        tr_ms = get_items('RepetitionTime')[::n_volumes]

        # get echo time(s). Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
        te_ms = get_items('EchoTime')
        if all(val is None for val in te_ms):
            te_ms = get_items('EffectiveEchoTime')
        te_ms = te_ms[::n_volumes]

        misc = {name: make_unique_tensor(get_items(tag)) for name, tag in MISC_TAGS.items()}

        return cls(
            resolution=resolution,
            fa=None if fa_deg is None else deg_to_rad(fa_deg),
            ti=None if ti_ms is None else ms_to_s(ti_ms),
            tr=None if tr_ms is None else ms_to_s(tr_ms),
            te=None if te_ms is None else ms_to_s(te_ms),
            _misc=misc,
        )

    def __repr__(self):
        """Representation method for IHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        out = f'Resolution [m/pixel]: {self.resolution!s}\nTE [s]: {te}\nTI [s]: {ti}\nFlip angle [rad]: {fa}.'
        return out
