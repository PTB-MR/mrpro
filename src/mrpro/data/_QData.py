"""MR quantitative data (QData) class."""

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

import torch

from mrpro.data import IHeader
from mrpro.data import KHeader
from mrpro.data import QHeader


class QData:
    """MR quantitative data (QData) class."""

    def __init__(self, data: torch.Tensor, header: KHeader | IHeader | QHeader) -> None:
        """Create QData object from a tensor and an arbitrary MRpro header.

        Parameters
        ----------
        data
            quantitative image data tensor with dimensions (all_other, coils, z, y, x).
        header
            MRpro header containing required meta data for the QHeader.
        """
        if isinstance(header, KHeader):
            self.header = QHeader.from_kheader(header)
        elif isinstance(header, IHeader):
            self.header = QHeader.from_iheader(header)
        elif isinstance(header, QHeader):
            self.header = header

        self.data = data
