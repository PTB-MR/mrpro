"""Create dicom image datasets."""

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

import os
from dataclasses import dataclass

import numpy as np
import pydicom
import pydicom._storage_sopclass_uids


@dataclass(slots=True)
class DicomImageData():

    filename: str | os.PathLike
    matrix_size: int = 256
    imref: np.ndarray | None = None

    def create_2d_file(self):
        """Create dicom image file."""

        # Create random image
        self.imref = np.random.randn(self.matrix_size, self.matrix_size).astype(np.uint16)

        # Metadata
        fileMeta = pydicom.Dataset()
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

        ds.BitsAllocated = 16
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsStored = 16
        ds.PixelData = self.imref.tobytes()

        # Save
        ds.save_as(self.filename, write_like_original=False)
