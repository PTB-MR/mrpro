"""Create 2D dicom image datasets for testing."""

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

from pathlib import Path

import numpy as np
import pydicom
import pydicom._storage_sopclass_uids
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
        size of image matrix along y, by default 128
    matrix_size_x
        size of image matrix along x, by default 256
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
        im_dim = SpatialDimension(z=1, y=matrix_size_y, x=matrix_size_x)
        self.imref = torch.abs(self.phantom.image_space(im_dim))
        self.imref = self.imref[0, 0, 0, ...]  # we can only store a 2D image in the dicom here
        self.imref /= torch.max(self.imref) * 2  # *2 to make sure we are well within uint16 range later on
        self.imref = torch.round(self.imref * 2**16)

        # Metadata
        fileMeta = pydicom.dataset.FileMetaDataset()
        fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
        fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Dataset
        ds = pydicom.Dataset()
        ds.file_meta = fileMeta

        # When accessing the data using ds.pixel_array pydicom will return an image with dimensions (rows columns).
        # According to the dicom standard rows corresponds to the vertical dimension (i.e. y) and columns corresponds
        # to the horizontal dimension (i.e. x)
        ds.Rows = self.imref.shape[0]
        ds.Columns = self.imref.shape[1]
        ds.NumberOfFrames = 1

        ds.PixelSpacing = [1, 1]  # in mm
        ds.SliceThickness = 1  # in mm

        elem = pydicom.DataElement(0x00191015, 'FD', [1.0, 2.0, 3.0])
        ds.add(elem)

        ds.FlipAngle = 15.0
        ds.EchoTime = te
        ds.RepetitionTime = 25.2

        ds.BitsAllocated = 16
        ds.PixelRepresentation = 0  # uint
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsStored = 16
        ds.PixelData = self.imref.numpy().astype(np.uint16).tobytes()

        # Save
        ds.save_as(self.filename, write_like_original=False)
