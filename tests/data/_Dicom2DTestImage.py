"""Create 2D dicom image datasets for testing."""

from pathlib import Path

import numpy as np
import pydicom
import torch
from mrpro.data import SpatialDimension
from mrpro.phantoms import EllipsePhantom


class Dicom2DTestImage:
    """2D Dicom image for testing.

    Parameters
    ----------
    filename
        full path and filename
    matrix_size_y
        size of image matrix along y
    matrix_size_x
        size of image matrix along x
    phantom
        phantom with different ellipses
    """

    def __init__(
        self,
        filename: str | Path,
        matrix_size_y: int = 128,
        matrix_size_x: int = 256,
        te: float = 3.7,
        phantom: EllipsePhantom | None = None,
    ):
        if not phantom:
            phantom = EllipsePhantom()

        self.filename: str | Path = filename
        self.matrix_size_y: int = matrix_size_y
        self.matrix_size_x: int = matrix_size_x
        self.te: float = te
        self.phantom: EllipsePhantom = phantom

        # Create image
        img_dimension = SpatialDimension(z=1, y=matrix_size_y, x=matrix_size_x)
        self.img_ref = torch.abs(self.phantom.image_space(img_dimension))
        self.img_ref = self.img_ref[0, 0, 0, ...]  # we can only store a 2D image in the dicom here
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

        # When accessing the data using dataset.pixel_array pydicom will return an image with dimensions (rows columns).
        # According to the dicom standard rows corresponds to the vertical dimension (i.e. y) and columns corresponds
        # to the horizontal dimension (i.e. x)
        dataset.Rows = self.img_ref.shape[0]
        dataset.Columns = self.img_ref.shape[1]
        dataset.NumberOfFrames = 1

        dataset.PixelSpacing = [1, 1]  # in mm
        dataset.SliceThickness = 1  # in mm

        elem = pydicom.DataElement(0x00191015, 'FD', [1.0, 2.0, 3.0])
        dataset.add(elem)

        dataset.FlipAngle = 15.0
        dataset.EchoTime = te
        dataset.RepetitionTime = 25.2

        dataset.BitsAllocated = 16
        dataset.PixelRepresentation = 0  # uint
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = 'MONOCHROME2'
        dataset.BitsStored = 16
        dataset.PixelData = self.img_ref.numpy().astype(np.uint16).tobytes()

        # Save
        dataset.save_as(self.filename, write_like_original=False)
