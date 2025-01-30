"""Create 2D dicom image datasets for testing."""

import datetime
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pydicom
import torch
from einops import repeat
from mrpro.data import Rotation, SpatialDimension
from mrpro.phantoms import EllipsePhantom
from mrpro.utils.unit_conversion import s_to_ms


class DicomTestImage:
    """Dicom image for testing.

    Parameters
    ----------
    filename
        full path and filename
    matrix_size_y
        size of image matrix along y
    matrix_size_x
        size of image matrix along x
    slice_orientation
        Orientation of slice. If None assume transversal orientation.
    slice_offset
        Slice offset from isocentre along slice_orientation. The number of slices is determined by the length of this
        parameter. If the length is greater than 1, the dicom will be saved as a 3D volume.
    te
        echo time
    time_after_rpeak
        time after R-peak
    phantom
        phantom with different ellipses
    series_description
        description of the DICOM series
    series_instance_uid
        UID identifying a series, i.e. a set of images which belong together (e.g. multiple slices).
        If None, a new UID is generated.
    """

    def __init__(
        self,
        filename: str | Path,
        matrix_size_y: int = 32,
        matrix_size_x: int = 32,
        slice_orientation: Rotation | None = None,
        slice_offset: float | Sequence[float] = 0.0,
        te: float = 0.037,
        time_after_rpeak: float = 0.333,
        phantom: EllipsePhantom | None = None,
        series_description: str = 'dicom_test_images',
        series_instance_uid: str | None = None,
    ):
        if not phantom:
            phantom = EllipsePhantom()

        self.filename: str | Path = filename
        self.matrix_size_y: int = matrix_size_y
        self.matrix_size_x: int = matrix_size_x
        if slice_orientation is None:
            transversal_orientation: Sequence[SpatialDimension] = [
                SpatialDimension(1, 0, 0),
                SpatialDimension(0, 1, 0),
                SpatialDimension(0, 0, 1),
            ]

        self.slice_orientation: Sequence[SpatialDimension] = (
            transversal_orientation if slice_orientation is None else slice_orientation.as_directions()
        )
        self.slice_offset: torch.Tensor = torch.atleast_1d(torch.as_tensor(slice_offset))
        self.te: float = te
        self.time_after_rpeak: float = time_after_rpeak
        self.phantom: EllipsePhantom = phantom
        self.series_description = series_description
        self.series_instance_uid = pydicom.uid.generate_uid() if series_instance_uid is None else series_instance_uid

        # Create image
        img_dimension = SpatialDimension(z=1, y=matrix_size_y, x=matrix_size_x)
        self.img_ref = torch.abs(self.phantom.image_space(img_dimension))
        self.img_ref = self.img_ref[0, 0, 0, ...]  # we can only store a 2D or 3D image in the dicom here
        self.img_ref /= torch.max(self.img_ref) * 2  # *2 to make sure we are well within uint16 range later on
        self.img_ref = torch.round(self.img_ref * 2**16)

        # Metadata
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Dataset
        dataset = pydicom.Dataset()
        dataset.file_meta = file_meta

        dataset.PatientName = 'Ellipse Phantom'
        dataset.PatientID = 'Ellipse001'
        dataset.PatientSex = 'O'
        dataset.Modality = 'MR'
        dataset.StudyDescription = 'MRpro'
        dataset.SeriesDate = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        dataset.SeriesTime = datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S.%f')
        dataset.SeriesDescription = series_description
        dataset.SeriesInstanceUID = self.series_instance_uid

        # When accessing the data using dataset.pixel_array pydicom will return an image with dimensions (rows columns).
        # According to the dicom standard rows corresponds to the vertical dimension (i.e. y) and columns corresponds
        # to the horizontal dimension (i.e. x)
        dataset.Rows = self.img_ref.shape[0]
        dataset.Columns = self.img_ref.shape[1]

        elem = pydicom.DataElement(0x00191015, 'FD', [1.0, 2.0, 3.0])
        dataset.add(elem)

        dataset.BitsAllocated = 16
        dataset.PixelRepresentation = 0  # uint
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = 'MONOCHROME2'
        dataset.BitsStored = 16

        nslices = len(self.slice_offset)
        readout_direction = np.asarray(self.slice_orientation[2].zyx[::-1])
        phase_direction = np.asarray(self.slice_orientation[1].zyx[::-1])
        slice_direction = np.asarray(self.slice_orientation[0].zyx[::-1])

        # Set image resolution to 2mm in-plane and 4mm through-plance
        inplane_resolution = 2
        slice_thickness = 4
        # Patient position in dicom defines the pixel with index (0,0). Start in isocentre (0,0)
        patient_position = (
            -readout_direction * inplane_resolution * dataset.Columns / 2
            - phase_direction * inplane_resolution * dataset.Rows / 2
        )

        if nslices > 1:
            dataset.MRAcquisitionType = '3D'
            dataset.NumberOfFrames = nslices
            dataset.PerFrameFunctionalGroupsSequence = pydicom.Sequence()

            plane_position_sequence = pydicom.Sequence()
            plane_position_sequence.append(pydicom.Dataset())
            plane_orientation_sequence = pydicom.Sequence()
            plane_orientation_sequence.append(pydicom.Dataset())
            mr_echo_sequence = pydicom.Sequence()
            mr_echo_sequence.append(pydicom.Dataset())
            pixel_measure_sequence = pydicom.Sequence()
            pixel_measure_sequence.append(pydicom.Dataset())
            mr_timing_parameters_sequence = pydicom.Sequence()
            mr_timing_parameters_sequence.append(pydicom.Dataset())

            for slice_idx in range(nslices):
                dataset.PerFrameFunctionalGroupsSequence.append(pydicom.Dataset())

                plane_position_sequence[0].ImagePositionPatient = (
                    patient_position + slice_direction * self.slice_offset[slice_idx].numpy()
                ).tolist()
                dataset.PerFrameFunctionalGroupsSequence[-1].PlanePositionSequence = deepcopy(plane_position_sequence)

                plane_orientation_sequence[0].ImageOrientationPatient = [*readout_direction, *phase_direction]
                dataset.PerFrameFunctionalGroupsSequence[-1].PlaneOrientationSequence = deepcopy(
                    plane_orientation_sequence
                )

                mr_echo_sequence[0].EffectiveEchoTime = s_to_ms(self.te)
                dataset.PerFrameFunctionalGroupsSequence[-1].MREchoSequence = deepcopy(mr_echo_sequence)

                pixel_measure_sequence[0].SliceThickness = slice_thickness
                pixel_measure_sequence[0].PixelSpacing = [inplane_resolution, inplane_resolution]
                dataset.PerFrameFunctionalGroupsSequence[-1].PixelMeasuresSequence = deepcopy(pixel_measure_sequence)

                mr_timing_parameters_sequence[0].FlipAngle = 15.0
                mr_timing_parameters_sequence[0].RepetitionTime = 25.2
                mr_timing_parameters_sequence[0].TriggerTime = s_to_ms(self.time_after_rpeak)
                dataset.PerFrameFunctionalGroupsSequence[-1].MRTimingAndRelatedParametersSequence = deepcopy(
                    mr_timing_parameters_sequence
                )

        else:
            dataset.MRAcquisitionType = '2D'
            dataset.NumberOfFrames = 1

            dataset.ImagePositionPatient = (patient_position + slice_direction * self.slice_offset.numpy()).tolist()
            dataset.ImageOrientationPatient = [*readout_direction, *phase_direction]
            dataset.EchoTime = s_to_ms(self.te)
            dataset.TriggerTime = s_to_ms(self.time_after_rpeak)
            dataset.PixelSpacing = [inplane_resolution, inplane_resolution]
            dataset.SliceThickness = slice_thickness
            dataset.FlipAngle = 15.0
            dataset.RepetitionTime = 25.2

        dataset.PixelData = (
            repeat(self.img_ref, 'x y -> nslices x y', nslices=nslices).numpy().astype(np.uint16).tobytes()
        )

        # Save
        dataset.save_as(self.filename, write_like_original=False)
