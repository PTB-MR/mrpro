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
from pydicom.dataset import Dataset
from pydicom.tag import Tag

from mrpro.data import Data
from mrpro.data import IHeader
from mrpro.data import KHeader


def _dcm_pixelarray_to_tensor(ds: Dataset) -> torch.Tensor:
    """Transform pixel array in dicom file to tensor.

    "Rescale intercept, (0028|1052), and rescale slope (0028|1053) are
    DICOM tags that specify the linear transformation from pixels in
    their stored on disk representation to their in memory
    representation.     U = m*SV + b where U is in output units, m is
    the rescale slope, SV is the stored value, and b is the rescale
    intercept." (taken from
    https://www.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/)
    """
    slope = float(ds.data_element('RescaleSlope').value) if 'RescaleSlope' in ds else 1.0

    intercept = float(ds.data_element('RescaleSlope').value) if 'RescaleIntercept' in ds else 0.0

    # Image data is 2D np.array of Uint16, which cannot directly be converted to tensor
    return slope * torch.as_tensor(ds.pixel_array.astype(np.complex64)) + intercept


@dataclasses.dataclass(slots=True, frozen=True)
class IData(Data):
    """MR image data (IData) class."""

    header: IHeader

    @classmethod
    def from_tensor_and_kheader(cls, data: torch.Tensor, kheader: KHeader) -> IData:
        """Create IData object from a tensor and a KHeader object.

        Parameters
        ----------
        data
            torch.Tensor containing image data with dimensions (broadcastable to) (other, coils, z, y, x).
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
        filename
            path to DICOM file.
        """
        ds = dcmread(filename)
        idata = _dcm_pixelarray_to_tensor(ds)[None, :]
        idata = rearrange(idata, '(other coils z) y x -> other coils z y x', other=1, coils=1, z=1)

        header = IHeader.from_dicom_list([ds])
        return cls(data=idata, header=header)

    @classmethod
    def from_dicom_folder(cls, foldername: str | Path, suffix: str | None = 'dcm') -> IData:
        """Read all DICOM files from a folder and return IData object.

        Parameters
        ----------
        foldername
            path to folder with DICOM files.
        suffix
            file extension (without period/full stop) to identify the DICOM files.
            If None, then all files in the folder are read in.
        """
        # Get files
        file_paths = list(Path(foldername).glob('*')) if suffix is None else list(Path(foldername).glob('*.' + suffix))

        if len(file_paths) == 0:
            raise ValueError(f'No dicom files with suffix {suffix} found in {foldername}')

        # Read in all files
        ds_list = [dcmread(filename) for filename in file_paths]

        # Ensure they all have the same orientation (same (0019, 1015) SlicePosition_PCS tag)
        def get_unique_slice_pos(slice_pos_tag: Tag = 0x00191015):
            if not ds_list[0].get_item(slice_pos_tag):
                return []
            else:
                sl_pos = [ds_list[0].get_item(slice_pos_tag).value]
                for ds in ds_list[1:]:
                    _val = ds.get_item(slice_pos_tag).value
                    if _val not in sl_pos and not isinstance(_val, bytes):
                        sl_pos.append(_val)
                return sl_pos

        if len(ds_list) > 1 and len(get_unique_slice_pos()) > 1:
            raise ValueError('Only dicoms with the same orientation can be read in.')

        # torch.stack is necessary otherwises mypy does not realize that rearrange yields Tensor from list[Tensor]
        idata = torch.stack([_dcm_pixelarray_to_tensor(ds) for ds in ds_list])
        idata = rearrange(idata, '(other coils z) y x -> other coils z y x', other=len(idata), coils=1, z=1)

        header = IHeader.from_dicom_list(ds_list)
        return cls(data=idata, header=header)
