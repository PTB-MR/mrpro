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

import dataclasses

import torch

from mrpro.data import IHeader
from mrpro.data import KHeader
from mrpro.data import QHeader


@dataclasses.dataclass(slots=True, frozen=True)
class QData:
    """MR quantitative data (QData) class."""

    header: QHeader
    data: torch.Tensor

    @classmethod
    def from_tensor_and_header(cls, data: torch.Tensor, header: KHeader | IHeader | QHeader) -> QData:
        """Create QData object from a tensor and an arbitrary MRpro header.

        Parameters
        ----------
        data  # ToDo: add which dimensions?
            torch.Tensor containing quantitative image data with dimensions (all_other, coils, z, x, y).
        kheader
            MRpro header (KHeader, IHeader or QHeader) containing required meta data for the QHeader.
        """
        if isinstance(header, KHeader):
            qheader = QHeader.from_kheader(header)
        elif isinstance(header, IHeader):
            qheader = QHeader.from_iheader(header)
        elif isinstance(header, QHeader):
            qheader = header
        return cls(header=qheader, data=data)
