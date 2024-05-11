"""Base class for data objects."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
from abc import ABC
from typing import Any

import torch

from mrpro.data._MoveDataMixin import MoveDataMixin


@dataclasses.dataclass(slots=True, frozen=True)
class Data(MoveDataMixin, ABC):
    """A general data class with field data and header."""

    data: torch.Tensor
    header: Any
