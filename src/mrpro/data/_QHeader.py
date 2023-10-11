"""MR quantitative data header (QHeader) dataclass."""

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

from mrpro.data import IHeader
from mrpro.data import KHeader
from mrpro.data import SpatialDimension


@dataclass(slots=True)
class QHeader:
    """MR quantitative data header."""

    # ToDo: decide which attributes to store in the header
    fov: SpatialDimension[float]

    @classmethod
    def from_iheader(cls, iheader: IHeader):
        """Create QHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """

        return cls(fov=iheader.fov)

    @classmethod
    def from_kheader(cls, kheader: KHeader):
        """Create QHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """

        return cls(fov=kheader.recon_fov)

    @classmethod
    def from_dicom(cls, dicom_dataset: Dataset):
        """Read DICOM file containing qMRI data and return QHeader object.

        Parameters
        ----------
        dicom_dataset
            Dataset object containing the DICOM file.
        """

        def getItems(name):  # Todo: move to utils and reuse logic in IHeader
            """Get all items with a given name from a pydicom dataset."""
            # iterall is recursive, so it will find all items with the given name
            return [item.value for item in dicom_dataset.iterall() if item.tag == Tag(name)]

        fov_x_mm = float(getItems('Rows')[0]) * getItems('PixelSpacing')[0][0]
        fov_y_mm = float(getItems('Columns')[0]) * getItems('PixelSpacing')[0][1]
        fov_z_mm = float(getItems('SliceThickness')[0])
        fov = SpatialDimension(fov_x_mm / 1000.0, fov_y_mm / 1000.0, fov_z_mm / 1000.0)
        return cls(fov=fov)
