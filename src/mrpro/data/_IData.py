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

from mrpro.data import IHeader
from mrpro.data import KHeader


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

    @staticmethod
    def _dcm_pixelarray_to_tensor(ds: Dataset) -> torch.Tensor:
        """Transform pixel array in dicom file to tensor.

        "Rescale intercept, (0028|1052), and rescale slope (0028|1053)
        are DICOM tags that specify the linear transformation from
        pixels in their stored on disk representation to their in memory
        representation.     U = m*SV + b where U is in output units, m
        is the rescale slope, SV is the stored value, and b is the
        rescale intercept." (taken from
        https://www.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/)
        """
        if 'RescaleSlope' in ds:
            slope = float(ds.data_element('RescaleSlope').value)
        else:
            slope = 1.0

        if 'RescaleIntercept' in ds:
            intercept = float(ds.data_element('RescaleSlope').value)
        else:
            intercept = 0.0

        # Image data is 2D np.array of Uint16, which cannot directly be converted to tensor
        return slope * torch.as_tensor(ds.pixel_array.astype(np.complex64)) + intercept

    @classmethod
    def from_single_dicom(cls, filename: str | Path) -> IData:
        """Read single DICOM file and return IData object.

        Parameters
        ----------
        filename:
            Path to DICOM file.
        """

        ds = dcmread(filename)
        idata = rearrange(
            IData._dcm_pixelarray_to_tensor(ds)[None, :], '(other coil z) y x -> other coil z y x', other=1, coil=1, z=1
        )

        header = IHeader.from_dicom_list([ds])
        return cls(data=idata, header=header)

    @classmethod
    def from_dicom_folder(cls, foldername: str | Path, suffix: str | None = 'dcm') -> IData:
        """Read all DICOM files from a folder and return IData object.

        Parameters
        ----------
        foldername:
            Path to folder with DICOM files.
        suffix:
            File extension (without period/full stop) to identify the DICOM files.
            If None, then all files in the folder are read in.
        """

        # Get files
        if suffix is None:  # Read all files in the folder
            filenames = list(Path(foldername).glob('*'))
        else:
            filenames = list(Path(foldername).glob('*.' + suffix))

        if len(filenames) == 0:
            raise ValueError(f'No dicom files with suffix {suffix} found in {foldername}')

        # Read in all files
        ds_list = [dcmread(filename) for filename in filenames]

        # Ensure they all have the same orientation (same (0019, 1015) SlicePosition_PCS tag)
        def get_unique_slice_pos():
            sl_pos = [ds_list[0][0x00191015].value]
            for ds in ds_list[1:]:
                if ds[0x00191015].value not in sl_pos:
                    sl_pos.append(ds[0x00191015].value)
            return sl_pos

        if len(ds_list) > 1 and len(get_unique_slice_pos()) > 1:
            raise ValueError('Only dicoms with the same orientation can be read in.')

        # torch.stack is necessary otherwises mypy does not realize that rearrange yields Tensor from list[Tensor]
        idata = torch.stack([IData._dcm_pixelarray_to_tensor(ds) for ds in ds_list])
        idata = rearrange(idata, '(other coil z) y x -> other coil z y x', other=len(idata), coil=1, z=1)

        header = IHeader.from_dicom_list(ds_list)
        return cls(data=idata, header=header)
