"""Create dicom image datasets for testing."""

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
from mrpro.utils import RandomGenerator
from mrpro.utils.unit_conversion import s_to_ms
from pydicom.dataset import set_pixel_data


class DicomTestImage:
    """Dicom image for testing."""

    def __init__(
        self,
        filename: str | Path,
        matrix_size_y: int = 32,
        matrix_size_x: int = 40,
        slice_orientation: Rotation | None = None,
        slice_offset: float | Sequence[float] = 0.0,
        cardiac_trigger_delay: float | None = None,
        te: float = 0.037,
        time_after_rpeak: float = 0.333,
        phantom: EllipsePhantom | None = None,
        series_description: str = 'dicom_test_images',
        series_instance_uid: str | None = None,
    ):
        """Initialize DicomTestImage.

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
            Slice offset from isocentre along slice_orientation. The number of slices is determined by the length of
            this parameter. If the length is greater than 1, the dicom will be saved as a 3D volume.
        cardiac_trigger_delay
            Cardiac trigger delay time. If set, the slice offset sequence is used for the cardiac phases,
            i.e. the number of slice_offset determines the number of cardiac phases.
            Offset values may be equal in this case.
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

        if not phantom:
            phantom = EllipsePhantom()

        self.filename = filename
        self.matrix_size_y = matrix_size_y
        self.matrix_size_x = matrix_size_x
        if slice_orientation is None:
            transversal_orientation: Sequence[SpatialDimension] = [
                SpatialDimension(1, 0, 0),
                SpatialDimension(0, 1, 0),
                SpatialDimension(0, 0, 1),
            ]

        self.slice_orientation: Sequence[SpatialDimension] = (
            transversal_orientation if slice_orientation is None else slice_orientation.as_directions()
        )
        self.slice_offset = torch.atleast_1d(torch.as_tensor(slice_offset))
        self.te = te
        self.time_after_rpeak = time_after_rpeak
        self.cardiac_trigger_delay = cardiac_trigger_delay
        self.phantom = phantom
        self.series_description = series_description
        self.series_instance_uid = pydicom.uid.generate_uid() if series_instance_uid is None else series_instance_uid
        dt = datetime.datetime.now(datetime.timezone.utc)

        # Create image
        img_dimension = SpatialDimension(z=1, y=matrix_size_y, x=matrix_size_x)
        self.img_ref = torch.abs(self.phantom.image_space(img_dimension))
        self.img_ref = self.img_ref[0, 0, 0, ...]  # we can only store a 2D or 3D image in the dicom here
        self.img_ref /= torch.max(self.img_ref)
        self.img_ref = torch.round(self.img_ref * (2**16 - 1))

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
        dataset.SeriesDate = dt.strftime('%Y%m%d')
        dataset.SeriesTime = dt.strftime('%H%M%S.%f')
        dataset.SeriesDescription = series_description
        dataset.SeriesInstanceUID = self.series_instance_uid

        elem = pydicom.DataElement(0x00191015, 'FD', [1.0, 2.0, 3.0])
        dataset.add(elem)

        n_slices = len(self.slice_offset)
        readout_direction = np.asarray(self.slice_orientation[2].zyx[::-1])
        phase_direction = np.asarray(self.slice_orientation[1].zyx[::-1])
        slice_direction = np.asarray(self.slice_orientation[0].zyx[::-1])

        # Set image resolution to 2mm in-plane and 4mm through-plance
        inplane_resolution = 2
        slice_thickness = 4
        # Patient position in dicom defines the pixel with index (0,0). Start in isocentre (0,0)
        patient_position = (
            -readout_direction * inplane_resolution * self.img_ref.shape[-1] / 2
            - phase_direction * inplane_resolution * self.img_ref.shape[-2] / 2
        )

        if n_slices > 1:
            dataset.MRAcquisitionType = '3D' if self.cardiac_trigger_delay is None else '2D'
            dataset.NumberOfFrames = n_slices
            dataset.PerFrameFunctionalGroupsSequence = pydicom.Sequence()

            frame_time_spacing_s = datetime.timedelta(seconds=0.6)
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

            for slice_idx in range(n_slices):
                dataset.PerFrameFunctionalGroupsSequence.append(pydicom.Dataset())

                if self.cardiac_trigger_delay is not None:
                    # In case of cardiac data, the in stack position remains unchanged
                    # Here a constant index of 1 is used, in practice this dependents on the slice index
                    dataset.PerFrameFunctionalGroupsSequence[-1].InStackPositionNumber = 1
                    dataset.PerFrameFunctionalGroupsSequence[-1].FrameReferenceDateTime = dt + datetime.timedelta(
                        seconds=self.cardiac_trigger_delay * slice_idx
                    )
                    dataset.PerFrameFunctionalGroupsSequence[-1].NominalCardiacTriggerDelayTime = s_to_ms(
                        self.cardiac_trigger_delay * slice_idx
                    )
                    dataset.PerFrameFunctionalGroupsSequence[-1].TemporalPositionIndex = slice_idx + 1
                else:
                    dataset.PerFrameFunctionalGroupsSequence[-1].InStackPositionNumber = slice_idx + 1
                    dataset.PerFrameFunctionalGroupsSequence[-1].FrameReferenceDateTime = (
                        dt + frame_time_spacing_s * slice_idx
                    )

                plane_position_sequence[0].ImagePositionPatient = (
                    patient_position + slice_direction * slice_idx * slice_thickness
                ).tolist()
                dataset.PerFrameFunctionalGroupsSequence[-1].PlanePositionSequence = deepcopy(plane_position_sequence)

                plane_orientation_sequence[0].ImageOrientationPatient = [*phase_direction, *readout_direction]
                dataset.PerFrameFunctionalGroupsSequence[-1].PlaneOrientationSequence = deepcopy(
                    plane_orientation_sequence
                )

                mr_echo_sequence[0].EffectiveEchoTime = s_to_ms(self.te)
                dataset.PerFrameFunctionalGroupsSequence[-1].MREchoSequence = deepcopy(mr_echo_sequence)

                pixel_measure_sequence[0].SliceThickness = slice_thickness
                pixel_measure_sequence[0].PixelSpacing = [inplane_resolution, inplane_resolution]
                dataset.PerFrameFunctionalGroupsSequence[-1].PixelMeasuresSequence = deepcopy(pixel_measure_sequence)

                mr_timing_parameters_sequence[0].FlipAngle = 15.0
                mr_timing_parameters_sequence[0].InversionTime = 3.0
                mr_timing_parameters_sequence[0].RepetitionTime = 25.2
                mr_timing_parameters_sequence[0].TriggerTime = s_to_ms(self.time_after_rpeak)
                dataset.PerFrameFunctionalGroupsSequence[-1].MRTimingAndRelatedParametersSequence = deepcopy(
                    mr_timing_parameters_sequence
                )

        else:
            dataset.MRAcquisitionType = '2D'

            dataset.ImagePositionPatient = (patient_position + slice_direction * self.slice_offset.numpy()).tolist()
            dataset.ImageOrientationPatient = [*phase_direction, *readout_direction]
            dataset.EchoTime = s_to_ms(self.te)
            dataset.InversionTime = 3.0
            dataset.TriggerTime = s_to_ms(self.time_after_rpeak)
            dataset.PixelSpacing = [inplane_resolution, inplane_resolution]
            dataset.SliceThickness = slice_thickness
            dataset.InStackPositionNumber = 1
            dataset.FlipAngle = 15.0
            dataset.RepetitionTime = 25.2
            dataset.FrameReferenceDateTime = dt

        random = RandomGenerator()
        noise = random.float32_tensor(size=(n_slices, matrix_size_x, matrix_size_y))

        # 'MONOCHROME2' means smallest value is black, largest value is white
        set_pixel_data(
            ds=dataset,
            arr=(repeat(self.img_ref, 'y x -> slices x y', slices=n_slices) + noise).numpy().astype(np.uint16),
            photometric_interpretation='MONOCHROME2',
            bits_stored=16,
        )

        # Save
        dataset.save_as(self.filename, enforce_file_format=True)
