"""MR quantitative data (QData) class."""

import dataclasses
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from pydicom import dcmread
from typing_extensions import Self

from mrpro.data.Data import Data
from mrpro.data.IHeader import IHeader
from mrpro.data.KHeader import KHeader
from mrpro.data.QHeader import QHeader


@dataclasses.dataclass(init=False, slots=True, frozen=True)
class QData(Data):
    """MR quantitative data (QData) class."""

    header: QHeader
    """Header describing quantitative data."""

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
    def from_single_dicom(cls, filename: str | Path) -> Self:
        """Read single DICOM file and return QData object.

        Parameters
        ----------
        filename
            path to DICOM file
        """
        dataset = dcmread(filename)
        # Image data is 2D np.array of Uint16, which cannot directly be converted to tensor
        qdata = torch.as_tensor(dataset.pixel_array.astype(np.complex64))
        qdata = repeat(qdata, 'y x -> other coils z y x', other=1, coils=1, z=1)
        header = QHeader.from_dicom(dataset)
        return cls(data=qdata, header=header)

    def __repr__(self):
        """Representation method for QData class."""
        try:
            device = str(self.device)
        except RuntimeError:
            device = 'mixed'
        out = (
            f'{type(self).__name__} with shape: {list(self.data.shape)!s} and dtype {self.data.dtype}\n'
            f'Device: {device}\nFOV [m]: {self.header.fov!s}.'
        )
        return out
