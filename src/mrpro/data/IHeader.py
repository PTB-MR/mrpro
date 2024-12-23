"""MR image data header (IHeader) dataclass."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from einops import repeat
from pydicom.dataset import Dataset
from pydicom.tag import Tag, TagType
from typing_extensions import Self

from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.remove_repeat import remove_repeat
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import deg_to_rad, mm_to_m, ms_to_s

from .AcqInfo import PhysiologyTimestamps

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

    fov: SpatialDimension[float]
    """Field of view [m]."""

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

    ImageIdx: ImageIdx = field(default_factory=ImageIdx)

    @classmethod
    def from_kheader(cls, kheader: KHeader) -> Self:
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """
        return cls(fov=kheader.recon_fov, te=kheader.te, ti=kheader.ti, fa=kheader.fa, tr=kheader.tr)

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

        def get_items_from_dicom_datasets(name: TagType) -> list:
            """Get list of items for all datasets in dicom_datasets."""
            return [get_item(ds, name) for ds in dicom_datasets]

        def get_float_items_from_dicom_datasets(name: TagType) -> list[float]:
            """Get float items from all dataset in dicom_datasets."""
            items = []
            for item in get_items_from_dicom_datasets(name):
                try:
                    items.append(float(item))
                except (TypeError, ValueError):
                    # None or invalid value
                    items.append(float('nan'))
            return items

        def as_5d_tensor(values: Sequence[float]) -> torch.Tensor:
            """Convert a list of values to a 5d tensor."""
            tensor = torch.as_tensor(values)
            tensor = repeat(tensor, 'values-> values 1 1 1 1')
            tensor = remove_repeat(tensor, 1e-12)
            return tensor

        def all_nan_to_none(tensor: torch.Tensor) -> torch.Tensor | None:
            """If all values are nan, return None."""
            if torch.isnan(tensor).all():
                return None
            return tensor

        fa = all_nan_to_none(deg_to_rad(as_5d_tensor(get_float_items_from_dicom_datasets('FlipAngle'))))
        ti = all_nan_to_none(ms_to_s(as_5d_tensor(get_float_items_from_dicom_datasets('InversionTime'))))
        tr = all_nan_to_none(ms_to_s(as_5d_tensor(get_float_items_from_dicom_datasets('RepetitionTime'))))

        te_list = get_float_items_from_dicom_datasets('EchoTime')
        if all(val is None for val in te_list):
            # if all 'EchoTime' entries are None, try 'EffectiveEchoTime',
            # which is used by some scanners
            te_list = get_float_items_from_dicom_datasets('EffectiveEchoTime')
        te = all_nan_to_none(ms_to_s(as_5d_tensor(te_list)))

        try:
            fov_x = mm_to_m(
                get_float_items_from_dicom_datasets('Rows')[0]
                * float(get_items_from_dicom_datasets('PixelSpacing')[0][0])
            )
        except (TypeError, ValueError):
            fov_x = float('nan')
        try:
            fov_y = mm_to_m(
                get_float_items_from_dicom_datasets('Columns')[0]
                * float(get_items_from_dicom_datasets('PixelSpacing')[0][1])
            )
        except (TypeError, ValueError):
            fov_y = float('nan')
        try:
            fov_z = mm_to_m(get_float_items_from_dicom_datasets('SliceThickness')[0])
        except (TypeError, ValueError):
            fov_z = float('nan')
        fov = SpatialDimension(fov_z, fov_y, fov_x)

        # Get misc parameters
        misc = {}
        for name in MISC_TAGS:
            misc[name] = as_5d_tensor(get_float_items_from_dicom_datasets(MISC_TAGS[name]))
        return cls(fov=fov, te=te, ti=ti, fa=fa, tr=tr, _misc=misc)

    def __repr__(self):
        """Representation method for IHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        out = f'FOV [m]: {self.fov!s}\n' f'TE [s]: {te}\nTI [s]: {ti}\nFlip angle [rad]: {fa}.'
        return out
