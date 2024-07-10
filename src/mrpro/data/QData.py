"""MR quantitative data (QData) class."""

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
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from pydicom import dcmread

from mrpro.data.Data import Data
from mrpro.data.IHeader import IHeader
from mrpro.data.KHeader import KHeader
from mrpro.data.QHeader import QHeader


@dataclasses.dataclass(init=False, slots=True, frozen=True)
class QData(Data):
    """MR quantitative data (QData) class."""

    header: QHeader

    def __init__(self, data: torch.Tensor, header: KHeader | IHeader | QHeader) -> None:
        """Create QData object from a tensor and an arbitrary MRpro header.

        Parameters
        ----------
        data
            quantitative image data tensor with dimensions (other, coils, z, y, x)
        header
            MRpro header containing required meta data for the QHeader
        """
        if isinstance(header, KHeader):
            qheader = QHeader.from_kheader(header)
        elif isinstance(header, IHeader):
            qheader = QHeader.from_iheader(header)
        elif isinstance(header, QHeader):
            qheader = header
        else:
            raise ValueError(f'Invalid header type: {type(header)}')

        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'header', qheader)

    @classmethod
    def from_single_dicom(cls, filename: str | Path) -> QData:
        """Read single DICOM file and return QData object.

        Parameters
        ----------
        filename
            path to DICOM file
        """
        dataset = dcmread(filename)
        # Image data is 2D np.array of Uint16, which cannot directly be converted to tensor
        qdata = torch.as_tensor(dataset.pixel_array.astype(np.complex64))
        qdata = rearrange(qdata[None, ...], '(other coils z) y x -> other coils z y x', other=1, coils=1, z=1)

        header = QHeader.from_dicom(dataset)
        return cls(data=qdata, header=header)

    def __repr__(self):
        """Representation method for QData class."""
        fov = self.header.fov if self.header.fov is not None else 'none'
        return f'QData with \nshape: {list(self.data.shape)!s}\nFOV: x={fov.x}, y={fov.y}, z={fov.z}.'
