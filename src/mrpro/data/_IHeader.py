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

from dataclasses import dataclass

from pydicom.dataset import Dataset
from pydicom.tag import Tag

from mrpro.data._KHeader import KHeader
from mrpro.data._SpatialDimension import SpatialDimension


@dataclass(slots=True)
class IHeader:
    """MR image data header."""

    # ToDo: decide which attributes to store in the header
    # ToDo: decide on types. Allow None ? Or use empty list ?
    fov: SpatialDimension[float] | None
    te: list[float] | None
    ti: list[float] | None
    fa: list[float] | None
    tr: list[float] | None

    @classmethod
    def from_kheader(cls, kheader: KHeader):
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """

        return cls(
            fov=kheader.encoding_fov,
            te=kheader.te,
            ti=kheader.ti,
            fa=kheader.fa,
            tr=kheader.tr,
        )

    @classmethod
    def from_dicom(cls, dicom_dataset: Dataset):
        """Read DICOM file and return IHeader object.

        Parameters
        ----------
        dicom_dataset
            Dataset object containing the DICOM file.
        """

        def getItems(name):
            """Get all items with a given name from a pydicom dataset."""
            # iterall is recursive, so it will find all items with the given name
            return [item.value for item in dicom_dataset.iterall() if item.tag == Tag(name)]

        # ToDo: Decide on how to handle missing values: Empty List or None
        fa = getItems('FlipAngle')
        ti = getItems('InversionTime')
        tr = getItems('RepetitionTime')
        # at least one dicom example has no 'EchoTime' but 'EffectiveEchoTime'
        te = getItems('EchoTime') or getItems('EffectiveEchoTime')
        fov = None  # ToDo: get from dicom_dataset
        return cls(fov=fov, te=te, ti=ti, fa=fa, tr=tr)
