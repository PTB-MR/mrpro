"""MR image data header (IHeader) dataclass."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from pydicom.dataset import Dataset
from pydicom.tag import Tag

from mrpro.data import KHeader
from mrpro.data import SpatialDimension

MISC_TAGS = {'TimeAfterStart': 0x00191016}


@dataclass(slots=True)
class IHeader:
    """MR image data header."""

    # ToDo: decide which attributes to store in the header
    fov: SpatialDimension[float]
    te: list[float]
    ti: list[float]
    fa: list[float]
    tr: list[float]
    misc: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_kheader(cls, kheader: KHeader) -> IHeader:
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """
        return cls(fov=kheader.recon_fov, te=kheader.te, ti=kheader.ti, fa=kheader.fa, tr=kheader.tr)

    @classmethod
    def from_dicom_list(cls, dicom_datasets: list[Dataset]) -> IHeader:
        """Read DICOM files and return IHeader object.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """

        def get_item(ds, name: str | Tag):
            """Get item with a given name or Tag from a pydicom dataset."""
            tag = Tag(name) if isinstance(name, str) else name  # find item via value name

            # iterall is recursive, so it will find all items with the given name
            found_item = [item.value for item in ds.iterall() if item.tag == tag]

            if len(found_item) == 0:
                return None
            elif len(found_item) == 1:
                return found_item[0]
            else:
                raise ValueError(f'Item {name} found {len(found_item)} times.')

        def get_items_from_all_dicoms(name: str | Tag):
            """Get list of items for all dataset objects in the list."""
            return [get_item(ds, name) for ds in dicom_datasets]

        def get_float_items_from_all_dicoms(name: str | Tag):
            """Convert items to float."""
            items = get_items_from_all_dicoms(name)
            return [float(val) if val is not None else None for val in items]

        def make_unique(values: list[float]):
            """If all the values are the same only return one."""
            if any(val is None for val in values):
                return []
            elif len(np.unique(values)) == 1:
                return [values[0]]
            else:
                return values

        fa = make_unique(get_float_items_from_all_dicoms('FlipAngle'))
        ti = make_unique(get_float_items_from_all_dicoms('InversionTime'))
        tr = make_unique(get_float_items_from_all_dicoms('RepetitionTime'))

        # get echo time(s). Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
        te_list = get_float_items_from_all_dicoms('EchoTime')
        if all(val is None for val in te_list):  # check if all entries are None
            te_list = get_float_items_from_all_dicoms('EffectiveEchoTime')
        te = make_unique(te_list)

        fov_x_mm = get_float_items_from_all_dicoms('Rows')[0] * float(get_items_from_all_dicoms('PixelSpacing')[0][0])
        fov_y_mm = get_float_items_from_all_dicoms('Columns')[0] * float(
            get_items_from_all_dicoms('PixelSpacing')[0][1],
        )
        fov_z_mm = get_float_items_from_all_dicoms('SliceThickness')[0]
        fov = SpatialDimension(fov_x_mm / 1000.0, fov_y_mm / 1000.0, fov_z_mm / 1000.0)

        # Get misc parameters
        misc = {}
        for name in MISC_TAGS:
            misc[name] = make_unique(get_float_items_from_all_dicoms(MISC_TAGS[name]))
        return cls(fov=fov, te=te, ti=ti, fa=fa, tr=tr, misc=misc)
