"""MR image data (IData) class."""

import dataclasses
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from pydicom import dcmread
from pydicom.dataset import Dataset
from typing_extensions import Self

from mrpro.data.Data import Data
from mrpro.data.IHeader import IHeader
from mrpro.data.KHeader import KHeader


def _dcm_pixelarray_to_tensor(dataset: Dataset) -> torch.Tensor:
    """Transform pixel array in dicom file to tensor.

    Rescale intercept, (0028|1052), and rescale slope (0028|1053) are
    DICOM tags that specify the linear transformation from pixels in
    their stored on disk representation to their in memory
    representation.     U = m*SV + b where U is in output units, m is
    the rescale slope, SV is the stored value, and b is the rescale
    intercept [RES]_.

    References
    ----------
    .. [RES] Rescale intercept and slope https://www.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
    """
    slope = (
        float(element.value)
        if 'RescaleSlope' in dataset and (element := dataset.data_element('RescaleSlope')) is not None
        else 1.0
    )
    intercept = (
        float(element.value)
        if 'RescaleIntercept' in dataset and (element := dataset.data_element('RescaleIntercept')) is not None
        else 0.0
    )

    # Image data is 2D np.array of Uint16, which cannot directly be converted to tensor
    return slope * torch.as_tensor(dataset.pixel_array.astype(np.complex64)) + intercept


@dataclasses.dataclass(slots=True, frozen=True)
class IData(Data):
    """MR image data (IData) class."""

    header: IHeader
    """Header for image data."""

    def rss(self, keepdim: bool = False) -> torch.Tensor:
        """Root-sum-of-squares over coils image data.

        Parameters
        ----------
        keepdim
            if `True`, the output tensor has the same number of dimensions as the data tensor, and the coil dimension is
            kept as a singleton dimension. If `False`, the coil dimension is removed.

        Returns
        -------
            image data tensor with shape `(..., 1, z, y, x)` if `keepdim` is `True`
            or `(..., z, y, x)` if keepdim is `False`.
        """
        coildim = -4
        return self.data.abs().square().sum(dim=coildim, keepdim=keepdim).sqrt()

    @classmethod
    def from_tensor_and_kheader(cls, data: torch.Tensor, header: KHeader) -> Self:
        """Create IData object from a tensor and a KHeader object.

        Parameters
        ----------
        data
            image data with dimensions (broadcastable to) `(other, coils, z, y, x)`.
        header
            MR raw data header containing required meta data for the image header.
        """
        iheader = IHeader.from_kheader(header)
        return cls(header=iheader, data=data)

    @classmethod
    def from_dicom_files(cls, filenames: Sequence[str | Path] | Generator[Path, None, None] | str | Path) -> Self:
        """Read multiple DICOM files and return IData object.

        DICOM images can be saved as single-frame or multi-frame images [DCMMF]_.

        If the DICOM files are single-frame, we treat each file separately and stack them along the `other` dimension.
        If the DICOM files are multi-frame and the MRAcquisitionType is ``3D`` we treat the frame dimension as the `z`
        dimension. Otherwise, we move the frame dimension to the `other` dimension. Multiple multi-frame DICOM
        images are stacked along an additional `other` dimension before the frame dimension.
        Providing the list of files sorted by filename usually leads to a reasonable sorting of the data.

        Parameters
        ----------
        filenames
            Filename or sequence of DICOM filenames.

        References
        ----------
        .. [DCMMF] https://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.7.6.16.html
        """
        if isinstance(filenames, str | Path):
            datasets = [dcmread(filenames)]
        else:
            datasets = [dcmread(filename) for filename in filenames]
        if not datasets:  # check datasets (not filenames) to allow for filenames to be a Generator
            raise ValueError('No dicom files specified')

        header = IHeader.from_dicom(*datasets)

        # Ensure that data has the same shape and can be stacked
        if not all(ds.pixel_array.shape == datasets[0].pixel_array.shape for ds in datasets):
            raise ValueError('Only dicom files with data of the same shape can be stacked.')

        data = torch.stack([_dcm_pixelarray_to_tensor(ds) for ds in datasets])

        # NumberofFrames (0028|0008) he total number of frames contained within a Multi-frame Image
        number_of_frames = [item.value for item in datasets[0].iterall() if item.tag == 0x00280008]

        if len(number_of_frames) > 0 and float(number_of_frames[0]) > 1:  # multi-frame data
            # MRAcquisitionType (0018|0023) is 1D/2D/3D
            mr_acquisition_type = [item.value for item in datasets[0].iterall() if item.tag == 0x00180023]

            if len(mr_acquisition_type) > 0 and mr_acquisition_type[0] == '3D':  # multi-frame 3D data
                data = repeat(data, 'other z y x -> other coils z y x', coils=1)
            else:  # multi-frame 2D data
                data = repeat(data, 'other frame y x -> other frame coils z y x', coils=1, z=1)
        else:  # single-frame data
            data = repeat(data, 'other y x -> other coils z y x', coils=1, z=1)

        return cls(data=data, header=header)

    @classmethod
    def from_dicom_folder(cls, foldername: str | Path, suffix: str | None = 'dcm') -> Self:
        """Read all DICOM files from a folder and return IData object.

        Parameters
        ----------
        foldername
            path to folder with DICOM files.
        suffix
            file extension (without period/full stop) to identify the DICOM files.
            If `None`, then all files in the folder are read in.
        """
        file_paths = list(Path(foldername).glob('*')) if suffix is None else list(Path(foldername).glob('*.' + suffix))
        if len(file_paths) == 0:
            raise ValueError(f'No dicom files with suffix {suffix} found in {foldername}')

        # Pass on sorted file list as order of dicom files is often the same as the required order
        return cls.from_dicom_files(filenames=sorted(file_paths))

    def __repr__(self):
        """Representation method for IData class."""
        try:
            device = str(self.device)
        except RuntimeError:
            device = 'mixed'
        out = (
            f'{type(self).__name__} with shape: {list(self.data.shape)!s} and dtype {self.data.dtype}\n'
            f'Device: {device}\n{self.header}'
        )
        return out
