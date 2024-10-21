"""MR image data header (IHeader) dataclass."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pydicom.dataset import Dataset
from pydicom.tag import Tag, TagType
from typing_extensions import Self

from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues

MISC_TAGS = {'TimeAfterStart': 0x00191016}


@dataclass(slots=True)
class IHeader(MoveDataMixin):
    """MR image data header."""

    # ToDo: decide which attributes to store in the header
    fov: SpatialDimension[float]
    """Field of view [m]."""

    te: torch.Tensor | None
    """Echo time [s]."""

    ti: torch.Tensor | None
    """Inversion time [s]."""

    fa: torch.Tensor | None
    """Flip angle [rad]."""

    tr: torch.Tensor | None
    """Repetition time [s]."""

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
        fov = SpatialDimension(fov_x_mm, fov_y_mm, fov_z_mm) / 1000  # convert to m

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
