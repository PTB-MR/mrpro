"""MR image data (IData) class."""

import warnings
from collections.abc import Generator, Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pydicom
import torch
from einops import rearrange, repeat
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels import set_pixel_data
from typing_extensions import Self

from mrpro.data.Dataclass import Dataclass
from mrpro.data.IHeader import IHeader
from mrpro.data.KHeader import KHeader
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.unit_conversion import m_to_mm, s_to_ms


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

    def to_dicom_folder(
        self,
        foldername: str | Path,
        series_description: str | None = None,
        reference_patient_table_position: SpatialDimension | None = None,
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
        """
        if not isinstance(foldername, Path):
            foldername = Path(foldername)
        foldername.mkdir(parents=True, exist_ok=False)

        # We save 3D image data in each dicom file. This can either be a full 3D volume, multiple slices (M2D) or
        # a combination of y and x image dimensions an one other dimension, e.g. multiple cardiac phases of a 2D image.
        mr_acquisition_type = '3D' if self.data.shape[-3] > 1 else '2D'
        frame_dimension = next((i for i in range(-3, -len(self.data.shape) - 1, -1) if self.data.shape[i] > 1), -3)
        number_of_frames = self.data.shape[frame_dimension]
        pattern_in = ['d' + str(i) for i in range(self.data.ndim)]
        pattern_out = pattern_in.copy()
        pattern_out[frame_dimension], pattern_out[-3] = pattern_out[-3], pattern_out[frame_dimension]
        dcm_idata = self.rearrange(' '.join(pattern_in) + '->' + ' '.join(pattern_out))

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
        dataset.StudyDescription = 'MRpro'
        import datetime

        dataset.SeriesDate = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        dataset.SeriesTime = datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S.%f')
        if series_description:
            dataset.SeriesDescription = series_description
            dataset.ProtocolName = series_description
        dataset.SeriesInstanceUID = pydicom.uid.generate_uid()

        dataset.PatientPosition = 'HFS'

        for file_index, other in enumerate(np.ndindex(dcm_idata.shape[:-3])):
            dcm_file_idata = dcm_idata[(*other, slice(None), slice(None), slice(None))]

            dataset.MRAcquisitionType = mr_acquisition_type
            dataset.PerFrameFunctionalGroupsSequence = pydicom.Sequence()

            plane_position_sequence = pydicom.Sequence()
            plane_position_sequence.append(Dataset())
            plane_orientation_sequence = pydicom.Sequence()
            plane_orientation_sequence.append(Dataset())
            mr_echo_sequence = pydicom.Sequence()
            mr_echo_sequence.append(Dataset())
            pixel_measure_sequence = pydicom.Sequence()
            pixel_measure_sequence.append(Dataset())
            mr_timing_parameters_sequence = pydicom.Sequence()
            mr_timing_parameters_sequence.append(Dataset())

            for frame in range(number_of_frames):
                dcm_frame_idata = dcm_file_idata[..., frame, :, :]
                if dcm_frame_idata.header.shape.numel() != 1:
                    raise ValueError('Only single image can be saved as a frame.')
                directions = dcm_frame_idata.header.orientation[0].as_directions()
                readout_direction = np.squeeze(np.asarray(directions[2].zyx[::-1]))
                phase_direction = np.squeeze(np.asarray(directions[1].zyx[::-1]))
                slice_direction = np.squeeze(np.asarray(directions[0].zyx[::-1]))
                position = np.squeeze(np.asarray(dcm_frame_idata.header.position.zyx[::-1]))

                dataset.PerFrameFunctionalGroupsSequence.append(Dataset())

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
                    image_position_patient += np.asarray(
                        np.squeeze(
                            (dcm_frame_idata.header.patient_table_position - reference_patient_table_position).zyx[::-1]
                        )
                    )

                plane_position_sequence[0].ImagePositionPatient = m_to_mm(image_position_patient.tolist())
                dataset.PerFrameFunctionalGroupsSequence[-1].PlanePositionSequence = deepcopy(plane_position_sequence)

                # The direction cosines of the first row and the first column with respect to the patient
                plane_orientation_sequence[0].ImageOrientationPatient = [*phase_direction, *readout_direction]
                dataset.PerFrameFunctionalGroupsSequence[-1].PlaneOrientationSequence = deepcopy(
                    plane_orientation_sequence
                )

                def unique_parameter(parameter: torch.Tensor | list[float], parameter_name: str) -> float | None:
                    """Return unique value of parameter tensor or list. Raise warning if not unique."""
                    unique_parameter = torch.unique(torch.as_tensor(parameter))
                    if unique_parameter.numel() > 1:
                        warnings.warn(
                            f'{parameter_name} is not unique. Using first value. To ensure all values are saved, '
                            'split data into correct subsets first.',
                            stacklevel=2,
                        )
                    return unique_parameter[0].item() if len(unique_parameter) > 0 else None

                if echo_time := unique_parameter(dcm_frame_idata.header.te, 'te'):
                    mr_echo_sequence[0].EchoTime = s_to_ms(echo_time)

                dataset.PerFrameFunctionalGroupsSequence[-1].MREchoSequence = deepcopy(mr_echo_sequence)

                pixel_measure_sequence[0].SliceThickness = m_to_mm(dcm_frame_idata.header.resolution[0].z)
                pixel_measure_sequence[0].PixelSpacing = m_to_mm(
                    [dcm_frame_idata.header.resolution[0].x, dcm_frame_idata.header.resolution[0].y]
                )
                dataset.PerFrameFunctionalGroupsSequence[-1].PixelMeasuresSequence = deepcopy(pixel_measure_sequence)

                if flip_angle := unique_parameter(dcm_frame_idata.header.fa, 'fa'):
                    mr_timing_parameters_sequence[0].FlipAngle = s_to_ms(flip_angle)
                if inversion_time := unique_parameter(dcm_frame_idata.header.ti, 'ti'):
                    mr_timing_parameters_sequence[0].InversionTime = s_to_ms(inversion_time)
                if repetition_time := unique_parameter(dcm_frame_idata.header.tr, 'tr'):
                    mr_timing_parameters_sequence[0].RepetitionTime = s_to_ms(repetition_time)
                dataset.PerFrameFunctionalGroupsSequence[-1].MRTimingAndRelatedParametersSequence = deepcopy(
                    mr_timing_parameters_sequence
                )

            # (frames, rows, columns) for multi-frame grayscale data
            pixel_data = dcm_file_idata.data.abs().cpu().numpy()
            pixel_data = pixel_data / pixel_data.max() * 2**16
            pixel_data = rearrange(pixel_data[0, 0, ...], 'frames y x -> frames x y')

            # 'MONOCHROME2' means smallest value is black, largest value is white
            set_pixel_data(
                ds=dataset, arr=pixel_data.astype(np.uint16), photometric_interpretation='MONOCHROME2', bits_stored=16
            )

            # Save
            dataset.save_as(foldername / f'im_mrpro_{np.prod(file_index)}.dcm', enforce_file_format=True)
