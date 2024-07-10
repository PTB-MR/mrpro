"""MR image data header (IHeader) dataclass."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.tag import TagType

from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension

MISC_TAGS = {'TimeAfterStart': 0x00191016}


@dataclass(slots=True)
class IHeader(MoveDataMixin):
    """MR image data header."""

    # ToDo: decide which attributes to store in the header
    fov: SpatialDimension[float]
    te: torch.Tensor | None
    ti: torch.Tensor | None
    fa: torch.Tensor | None
    tr: torch.Tensor | None
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
    def from_dicom_list(cls, dicom_datasets: Sequence[Dataset]) -> IHeader:
        """Read DICOM files and return IHeader object.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """

        def get_item(dataset: Dataset, name: str | TagType):
            """Get item with a given name or Tag from a pydicom dataset."""
            tag = Tag(name) if isinstance(name, str) else name  # find item via value name

            # iterall is recursive, so it will find all items with the given name
            found_item = [item.value for item in dataset.iterall() if item.tag == tag]

            if len(found_item) == 0:
                return None
            elif len(found_item) == 1:
                return found_item[0]
            else:
                raise ValueError(f'Item {name} found {len(found_item)} times.')

        def get_items_from_all_dicoms(name: str | TagType):
            """Get list of items for all dataset objects in the list."""
            return [get_item(ds, name) for ds in dicom_datasets]

        def get_float_items_from_all_dicoms(name: str | TagType):
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

        fa = make_unique_tensor(get_float_items_from_all_dicoms('FlipAngle'))
        ti = make_unique_tensor(get_float_items_from_all_dicoms('InversionTime'))
        tr = make_unique_tensor(get_float_items_from_all_dicoms('RepetitionTime'))

        # get echo time(s). Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
        te_list = get_float_items_from_all_dicoms('EchoTime')
        if all(val is None for val in te_list):  # check if all entries are None
            te_list = get_float_items_from_all_dicoms('EffectiveEchoTime')
        te = make_unique_tensor(te_list)

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
        fov = self.fov if self.fov is not None else 'none'
        te = [str(torch.round(te, decimals=4)) for te in self.te] if self.te is not None else 'none'
        ti = [str(torch.round(ti, decimals=4)) for ti in self.ti] if self.ti is not None else 'none'
        fa = [str(torch.round(fa, decimals=4)) for fa in self.fa] if self.fa is not None else 'none'
        out = f'FOV: x={fov.x}, y={fov.y}, z={fov.z}\n' f'TE: {te}\nTI: {ti}\nflip angle: {fa}.'
        return out
