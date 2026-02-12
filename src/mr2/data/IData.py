"""MR image data (IData) class."""

import datetime
import re
import warnings
from collections.abc import Generator, Sequence
from pathlib import Path

import nibabel
import numpy as np
import pydicom
import torch
from einops import rearrange, repeat
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels import set_pixel_data
from typing_extensions import Self

from mr2.data.Dataclass import Dataclass
from mr2.data.IHeader import IHeader, get_items
from mr2.data.KHeader import KHeader
from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.reshape import unsqueeze_right
from mr2.utils.summarize import summarize_values
from mr2.utils.unit_conversion import m_to_mm, rad_to_deg, s_to_ms


def _dcm_pixelarray_to_tensor(dataset: Dataset) -> torch.Tensor:
    """Transform pixel array in dicom file to tensor.

    Rescale intercept, (0028|1052), and rescale slope (0028|1053) are
    DICOM tags that specify the linear transformation from pixels in
    their stored on disk representation to their in memory
    representation. U = m*SV + b where U is in output units, m is
    the rescale slope, SV is the stored value, and b is the rescale
    intercept [RES]_.

    Rescale intercept and rescale slope can be defined for each frame in a 3D/M2D dicom file, as a global value or
    not at all.

    References
    ----------
    .. [RES] Rescale intercept and slope https://www.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
    """
    slope = sl if (sl := get_items([dataset], 'RescaleSlope', lambda x: float(x))) else [1.0]
    slope_tensor = unsqueeze_right(torch.as_tensor(slope), dataset.pixel_array.ndim - 1)
    intercept = ic if (ic := get_items([dataset], 'RescaleIntercept', lambda x: float(x))) else [0.0]
    intercept_tensor = unsqueeze_right(torch.as_tensor(intercept), dataset.pixel_array.ndim - 1)

    # Image data is 2D or 3D np.array of Uint16, which cannot directly be converted to tensor
    return slope_tensor * torch.as_tensor(dataset.pixel_array.astype(np.complex64)) + intercept_tensor


def _natural_key(s: str | Path) -> list[int | str]:
    """Key for natural sorting of strings with numbers.

    Example:
        >>> strings = ['img_1.dcm', 'img_10.dcm', 'img_2.dcm']
        >>> sorted(strings, key=_natural_key)
        ['img_1.dcm', 'img_2.dcm', 'img_10.dcm']
    """
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


class IData(Dataclass):
    """MR image data (IData) class."""

    data: torch.Tensor
    """Tensor containing image data with dimensions `(*other, coils, z, y, x)`."""

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
            # Use natsort to ensure correct order of filenames like img_1.dcm, img_2.dcm, ..., img_10.dcm
            datasets = [dcmread(filename) for filename in sorted(filenames, key=_natural_key)]
        if not datasets:  # check datasets (not filenames) to allow for filenames to be a Generator
            raise ValueError('No dicom files specified')

        header = IHeader.from_dicom(*datasets)

        # Ensure that data has the same shape and can be stacked
        if not all(ds.pixel_array.shape == datasets[0].pixel_array.shape for ds in datasets):
            raise ValueError('Only dicom files with data of the same shape can be stacked.')

        data = torch.stack([_dcm_pixelarray_to_tensor(ds) for ds in datasets])

        # NumberofFrames (0028|0008): The total number of frames contained within a Multi-frame Image
        number_of_frames = [item.value for ds in datasets for item in ds.iterall() if item.tag == 0x00280008]

        if len(number_of_frames) > 0 and float(number_of_frames[0]) > 1:  # multi-frame data
            # MRAcquisitionType (0018|0023) is 1D/2D/3D
            mr_acquisition_type = [item.value for item in datasets[0].iterall() if item.tag == 0x00180023]

            if len(mr_acquisition_type) > 0 and mr_acquisition_type[0] == '3D':  # multi-frame 3D data
                data = repeat(data, 'other z x y -> other coils z y x', coils=1)
            else:  # multi-frame 2D data, rearrange data and header
                if data.shape[0] == 1:  # if other dimension is singleton, there is only frame dimension
                    data = repeat(data, '1 frame x y -> frame coils z y x', coils=1, z=1)
                else:
                    data = repeat(data, 'other frame x y -> other frame coils z y x', coils=1, z=1)
                    header = header.rearrange('other coils frame x y -> other frame coils z y x', z=1)
        else:  # single-frame data
            data = repeat(data, 'other x y -> other coils z y x', coils=1, z=1)

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

    def to_nifti(self, filename: str | Path, magnitude_only: bool = False) -> None:
        """Save image data as NIFTI file.

        Parameters
        ----------
        filename
            filename / path of the NIFTI file.
        magnitude_only
            if `True`, only the magnitude of the image data is saved as a NIFTI1 file,
            otherwise the complex image data is saved as a NIFTI2 file.
            Many software packages do not support complex data.
        """
        orientation = self.header.orientation.mean().as_matrix()
        position = torch.stack([p.mean() for p in self.header.position.zyx])
        affine_zyx = torch.cat(
            [torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.cat([position[:, None], orientation], 1)], 0
        )
        affine = affine_zyx.flip([0, 1])
        data = rearrange(self.data, '... other coils z y x-> x y z 1 other (...) coils')
        if magnitude_only:
            nifti = nibabel.nifti2.Nifti1Image(data.abs().numpy(), affine.numpy(), dtype=np.float32)
        else:
            nifti = nibabel.nifti2.Nifti2Image(data.numpy(), affine.numpy(), dtype=np.complex64)

        nifti.header['pixdim'][1:4] = [self.header.resolution.x, self.header.resolution.y, self.header.resolution.z]
        description = (
            f'TE={summarize_values(self.header.te)}ms; '
            f'TI={summarize_values(self.header.ti)}ms; '
            f'TR={summarize_values(self.header.tr)}ms; '
            f'FA={summarize_values(self.header.fa)}rad'
        )
        nifti.header['descrip'] = description.encode('utf-8')
        nifti.to_filename(filename)

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

    def to_dicom_folder(
        self,
        foldername: str | Path,
        *,
        series_description: str | None = None,
        reference_patient_table_position: SpatialDimension | None = None,
        rescale_slope: float = 1.0,
        rescale_intercept: float = 0.0,
        normalize_data: bool = True,
    ) -> None:
        """Write the image data to DICOM files in a folder.

        The data is always saved in a multi-frame DICOM files.

        Parameters
        ----------
        foldername
            Path to folder for DICOM files.
        series_description
            String to be saved as the series description in the DICOM files.
        reference_patient_table_position
            If provided, the image position is calculated relative to this table positiion. This ensures that the
            image position is consistent across different scans even if the patient table has moved.
        rescale_slope
            Slope of linear scaling of data. Data is save as (data - intercept)/slope.
        rescale_intercept
            Intercept of linear scaling of data. Data is save as (data - intercept)/slope.
        normalize_data
            Normalize data prior to applying scaling.
        """
        if not isinstance(foldername, Path):
            foldername = Path(foldername)
        foldername.mkdir(parents=True, exist_ok=False)

        # We save 3D image data in each dicom file. This can either be a full 3D volume, multiple slices (M2D) or
        # a combination of y and x image dimensions an one other dimension, e.g. multiple cardiac phases of a 2D image.
        mr_acquisition_type = '3D' if self.data.shape[-3] > 1 else '2D'
        frame_dimension = next((i for i in range(-3, -len(self.data.shape) - 1, -1) if self.data.shape[i] > 1), -3)
        number_of_frames = self.data.shape[frame_dimension]
        dcm_idata = self.swapdims(frame_dimension, -3)

        # Metadata
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Dataset
        dataset = pydicom.Dataset()
        dataset.file_meta = file_meta

        dataset.PatientName = 'Unknown'
        dataset.PatientID = 'Unknown'
        dataset.PatientSex = 'O'
        dataset.Modality = 'MR'
        dataset.StudyDescription = 'mrtwo'

        timestamp = self.header.datetime or datetime.datetime.now(datetime.timezone.utc)
        dataset.SeriesDate = timestamp.strftime('%Y%m%d')
        dataset.SeriesTime = timestamp.strftime('%H%M%S.%f')
        if series_description:
            dataset.SeriesDescription = series_description
            dataset.ProtocolName = series_description
        dataset.SeriesInstanceUID = pydicom.uid.generate_uid()

        dataset.PatientPosition = 'HFS'

        for file_index, other in enumerate(np.ndindex(dcm_idata.shape[:-3])):
            dcm_file_idata = dcm_idata[(*other, slice(None), slice(None), slice(None))]

            dataset.MRAcquisitionType = mr_acquisition_type
            dataset.PerFrameFunctionalGroupsSequence = pydicom.Sequence()

            for frame in range(number_of_frames):
                dcm_frame_idata = dcm_file_idata[..., frame, :, :]
                frame_info = Dataset()
                if dcm_frame_idata.header.shape.numel() != 1:
                    raise ValueError('Only single image can be saved as a frame.')
                directions = dcm_frame_idata.header.orientation[0].as_directions()
                readout_direction = torch.stack(directions[2].zyx[::-1]).squeeze().numpy()
                phase_direction = torch.stack(directions[1].zyx[::-1]).squeeze().numpy()
                slice_direction = torch.stack(directions[0].zyx[::-1]).squeeze().numpy()
                position = torch.stack(dcm_frame_idata.header.position.zyx[::-1]).squeeze().numpy()

                image_position_patient = (
                    position
                    - readout_direction * dcm_frame_idata.header.resolution.x * dcm_frame_idata.data.shape[-1] / 2
                    - phase_direction * dcm_frame_idata.header.resolution.y * dcm_frame_idata.data.shape[-2] / 2
                )
                if mr_acquisition_type == '3D':
                    image_position_patient = (
                        image_position_patient
                        + slice_direction * dcm_frame_idata.header.resolution.z * (-number_of_frames / 2 + frame)
                    )
                if reference_patient_table_position:
                    image_position_patient += np.squeeze(
                        torch.stack(
                            (dcm_frame_idata.header.patient_table_position - reference_patient_table_position).zyx[::-1]
                        ).numpy()
                    )
                position_ds = Dataset()
                position_ds.ImagePositionPatient = m_to_mm(image_position_patient.tolist())
                frame_info.PlanePositionSequence = pydicom.Sequence([position_ds])

                # According to the dicom manual, ImageOrientationPatient describes:
                # "The direction cosines of the first row and the first column with respect to the patient."
                # This would suggest that the first direction should be the readout direction, the second the
                # phase direction. Nevertheless, for all our data the two directions have to be swapped to achieve
                # the correct orientation. Any help welcome in solving this!
                orientation_ds = Dataset()
                orientation_ds.ImageOrientationPatient = [*phase_direction, *readout_direction]
                frame_info.PlaneOrientationSequence = pydicom.Sequence([orientation_ds])

                def get_singleton(parameter: torch.Tensor | list[float], parameter_name: str) -> float | None:
                    """Return unique value of parameter tensor or list. Raise warning if not unique."""
                    unique_values = torch.unique(torch.as_tensor(parameter))
                    if unique_values.numel() > 1:
                        warnings.warn(
                            f'{parameter_name} is not singleton. Using first value. To ensure all values are saved, '
                            'split data into correct subsets first.',
                            stacklevel=2,
                        )
                    return unique_values[0].item() if len(unique_values) > 0 else None

                echo_ds = Dataset()
                if (echo_time := get_singleton(dcm_frame_idata.header.te, 'te')) is not None:
                    echo_ds.EchoTime = s_to_ms(echo_time)
                frame_info.MREchoSequence = pydicom.Sequence([echo_ds])

                pixel_measures = Dataset()
                resolution = dcm_frame_idata.header.resolution[0]
                pixel_measures.SliceThickness = m_to_mm(resolution.z)
                pixel_measures.PixelSpacing = m_to_mm([resolution.x, resolution.y])
                frame_info.PixelMeasuresSequence = pydicom.Sequence([Dataset(pixel_measures)])

                timing_parameters = Dataset()
                if (flip_angle := get_singleton(dcm_frame_idata.header.fa, 'fa')) is not None:
                    timing_parameters.FlipAngle = rad_to_deg(flip_angle)
                if (inversion_time := get_singleton(dcm_frame_idata.header.ti, 'ti')) is not None:
                    timing_parameters.InversionTime = s_to_ms(inversion_time)
                if (repetition_time := get_singleton(dcm_frame_idata.header.tr, 'tr')) is not None:
                    timing_parameters.RepetitionTime = s_to_ms(repetition_time)
                frame_info.MRTimingAndRelatedParametersSequence = pydicom.Sequence([timing_parameters])

                pixel_value_transformation = Dataset()
                pixel_value_transformation.RescaleSlope = rescale_slope
                pixel_value_transformation.RescaleIntercept = rescale_intercept
                pixel_value_transformation.RescaleType = 'US'
                frame_info.PixelValueTransformationSequence = pydicom.Sequence([pixel_value_transformation])

                dataset.PerFrameFunctionalGroupsSequence.append(frame_info)

            # (frames, rows, columns) for multi-frame grayscale data
            pixel_data = dcm_file_idata.data.abs().cpu().numpy()
            pixel_data = pixel_data / pixel_data.max() * (2**16 - 1) if normalize_data else pixel_data
            pixel_data = (pixel_data - rescale_intercept) / rescale_slope
            pixel_data = rearrange(pixel_data[0, 0, ...], 'frames y x -> frames x y')

            if np.any(clipped_idx := (pixel_data < 0) | (pixel_data > 65535)):
                clipped_data = pixel_data[clipped_idx]
                warnings.warn(
                    'Values outside of the uint16 range will be clipped. '
                    + f'Data range [{clipped_data.min()} - {clipped_data.max()}]',
                    stacklevel=2,
                )

            # 'MONOCHROME2' means smallest value is black, largest value is white
            set_pixel_data(
                ds=dataset, arr=pixel_data.astype(np.uint16), photometric_interpretation='MONOCHROME2', bits_stored=16
            )

            dataset.save_as(foldername / f'im_mr2_{np.prod(file_index)}.dcm', enforce_file_format=True)
