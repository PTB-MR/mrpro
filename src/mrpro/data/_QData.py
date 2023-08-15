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
from pathlib import Path

import pydicom
import torch

from mrpro.data._IHeader import IHeader


@dataclasses.dataclass(slots=True, frozen=True)
class QData:
    """MR quantitative data (QData) class."""
    
    header: IHeader
    data: torch.Tensor
    
    @classmethod
    def from_kdata(cls):
        pass
    
    @classmethod
    def from_idata(cls):
        pass
    
    @classmethod
    def from_ismrmrd(cls, fpath: str | Path) -> QData:
        data = torch.zeros(1)
        header = IHeader(None, None, None, None, None)
        
        return cls(data=data, header=header)
    
    @classmethod
    def from_nifty(cls, fpath: str | Path) -> QData:
        data = torch.zeros(1)
        header = IHeader(None, None, None, None, None)
        
        return cls(data=data, header=header)
    
    @classmethod
    def from_dicom(cls, fpath: str | Path) -> QData:
        data = torch.zeros(1)
        header = IHeader(None, None, None, None, None)
        
        return cls(data=data, header=header)