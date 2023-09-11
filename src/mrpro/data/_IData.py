"""MR image data (IData) class."""

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
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from pydicom import dcmread

from mrpro.data._IHeader import IHeader
from mrpro.data._KHeader import KHeader


@dataclasses.dataclass(slots=True, frozen=True)
class IData:
    """MR image data (IData) class."""

    header: IHeader
    data: torch.Tensor

    @classmethod
    def from_tensor_and_kheader(cls, data: torch.Tensor, kheader: KHeader) -> IData:
        """Create IData object from a tensor and a KHeader object.

        Parameters
        ----------
        data
            torch.Tensor containing image data with dimensions (broadcastable to) (all_other, coils, z, x, y).
        kheader
            MR raw data header (KHeader) containing required meta data for the image header (IHeader).
        """
        header = IHeader.from_kheader(kheader)
        return cls(header=header, data=data)

    @classmethod
    def from_single_dicom(cls, filename: str | Path) -> IData:
        """Read single DICOM file and return IData object.

        Parameters
        ----------
        filename:
            Path to DICOM file.
        """

        ds = dcmread(filename)
        # Image data is 2D np.array of Uint16, which cannot directly be converted to tensor
        idata = torch.as_tensor(ds.pixel_array.astype(np.complex64))
        idata = rearrange(idata[None, ...], '(other coil z) x y -> other coil z y x', other=1, coil=1, z=1)

        header = IHeader.from_dicom(ds)
        return cls(data=idata, header=header)
