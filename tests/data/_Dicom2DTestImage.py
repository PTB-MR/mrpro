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

from mrpro.phantoms import EllipsePhantom


class Dicom2DTestImage:
    """2D Dicom image for testing.

    Parameters
    ----------
    filename
        full path and filename
    matrix_size
        size of image matrix, by default 256
    phantom
        phantom with different ellipses
    """

    def __init__(self, filename: str | Path, matrix_size: int = 256, phantom: EllipsePhantom = EllipsePhantom()):
        self.filename: str | Path = filename
        self.matrix_size: int = matrix_size
        self.phantom: EllipsePhantom = phantom

        # Create image
        self.imref = self.phantom.image_space(matrix_size, matrix_size)
        self.imref = np.abs(self.imref * 2**16).astype(np.uint16)

        # Metadata
        fileMeta = pydicom.dataset.FileMetaDataset()
        fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
        fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Dataset
        ds = pydicom.Dataset()
        ds.file_meta = fileMeta

        ds.Rows = self.imref.shape[0]
        ds.Columns = self.imref.shape[1]
        ds.NumberOfFrames = 1

        ds.PixelSpacing = [1, 1]  # in mm
        ds.SliceThickness = 1  # in mm

        ds.FlipAngle = 15.0
        ds.EchoTime = 3.7
        ds.RepetitionTime = 25.2

        ds.BitsAllocated = 16
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsStored = 16
        ds.PixelData = self.imref.tobytes()

        # Save
        ds.save_as(self.filename, write_like_original=False)
