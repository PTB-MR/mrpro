"""MR image data (IData) class."""

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

from __future__ import annotations

from pathlib import Path

import torch

from mrpro.data._IHeader import IHeader


class IData:
    def __init__(self, header: IHeader, data: torch.Tensor) -> None:
        self._header: IHeader = header
        self._data: torch.Tensor = data

    @classmethod
    def from_dicom(cls, file_path: str | Path) -> IData:
        """Read DICOM file(s) and return IData object.

        Parameters
        ----------
        file_path:
            Path to DICOM file(s).
        """
        pass
