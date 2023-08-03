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

from pathlib import Path

import pydicom
import torch

from mrpro.data._IHeader import IHeader
from mrpro.data._KHeader import KHeader


class IData:
    def __init__(self, header: IHeader, data: torch.Tensor):
        """MR image data (IData) class.

        Parameters
        ----------
        header
            Image header containing meta data about the image.
        data
            Image data as complex torch.Tensor with shape (all_other, coils, z, x, y).
        """
        self._header: IHeader = header
        self._data: torch.Tensor = data

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def header(self) -> IHeader:
        return self._header

    @classmethod
    def from_tensor_and_kheader(cls, data: torch.Tensor, kheader: KHeader) -> IData:
        """Create IData object from a tensor and a KHeader object.

        Parameters
        ----------
        data
            torch.Tensor containing image data with dimensions (broadcastable to) (all_other, coils, z, x, y).
        kheader
            MR raw data header (KHeader) containing required meta data for the image header (IHeder).
        """
        header = IHeader.from_kheader(kheader)
        return cls(header=header, data=data)

    @classmethod
    def from_single_dicom(cls, fpath: str | Path) -> IData:
        """Read single DICOM file and return IData object.

        Parameters
        ----------
        fpath:
            Path to DICOM file.
        """

        def rget_item(
            dataset: pydicom.Dataset,
            tag: pydicom.tag.Tag,
        ) -> pydicom.DataElement:
            for ds_element in dataset:
                if ds_element.tag == tag:
                    return ds_element
                if ds_element.VR == 'SQ':  # if sequence
                    for seq_element in ds_element:
                        ret = rget_item(seq_element, tag)
                        if ret is not None and ret.tag == tag:
                            return ret

        fpath = Path(fpath)
        if not fpath.is_file():
            raise FileNotFoundError(f'File {fpath} not found.')

        ds = pydicom.dcmread(fpath)

        # get parameters using DICOM tags (https://www.dicomlibrary.com/dicom/dicom-tags/)
        # ToDo: find a cleaner and faster solution
        # ToDo: check wheter to use (0x0018, 0x9082) or (0x0018, 0x0080) for TE
        _tr = rget_item(ds, pydicom.tag.Tag((0x0018, 0x0080)))
        _te = rget_item(ds, pydicom.tag.Tag((0x0018, 0x9082)))
        _ti = rget_item(ds, pydicom.tag.Tag((0x0018, 0x0082)))
        _fa = rget_item(ds, pydicom.tag.Tag((0x0018, 0x1314)))
        # ToDo: check cases with multiple TE/TI etc...
        tr = None if not _tr else _tr.value
        te = None if not _te else _te.value
        ti = None if not _ti else _ti.value
        fa = None if not _fa else _fa.value

        # ToDo: bring into correct shape
        data = ds.pixel_array

        return cls(data=data, header=IHeader(None, te, ti, fa, tr))
