"""Class for encoding limits."""

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
from dataclasses import dataclass

from ismrmrd.xsd.ismrmrdschema.ismrmrd import encodingLimitsType
from ismrmrd.xsd.ismrmrdschema.ismrmrd import limitType


@dataclass(slots=True)
class Limits:
    """Limits dataclass with min, max, and center attributes."""

    min: int = 0
    max: int = 0
    center: int = 0

    @classmethod
    def from_ismrmrd(cls, limitType: limitType) -> Limits:
        if limitType is None:
            return cls()
        return cls(*dataclasses.astuple(limitType))

    @property
    def length(self) -> int:
        return self.max - self.min + 1


@dataclass(slots=True)
class EncodingLimits:
    """Encoding limits dataclass with limits for each attribute."""

    # replace with default with dataclasses.field(default_factory=Limits)
    kspace_encoding_step_0: Limits = dataclasses.field(default_factory=Limits)
    kspace_encoding_step_1: Limits = dataclasses.field(default_factory=Limits)
    kspace_encoding_step_2: Limits = dataclasses.field(default_factory=Limits)
    average: Limits = dataclasses.field(default_factory=Limits)
    slice: Limits = dataclasses.field(default_factory=Limits)
    contrast: Limits = dataclasses.field(default_factory=Limits)
    phase: Limits = dataclasses.field(default_factory=Limits)
    repetition: Limits = dataclasses.field(default_factory=Limits)
    set: Limits = dataclasses.field(default_factory=Limits)
    segment: Limits = dataclasses.field(default_factory=Limits)
    user_0: Limits = dataclasses.field(default_factory=Limits)
    user_1: Limits = dataclasses.field(default_factory=Limits)
    user_2: Limits = dataclasses.field(default_factory=Limits)
    user_3: Limits = dataclasses.field(default_factory=Limits)
    user_4: Limits = dataclasses.field(default_factory=Limits)
    user_5: Limits = dataclasses.field(default_factory=Limits)
    user_6: Limits = dataclasses.field(default_factory=Limits)
    user_7: Limits = dataclasses.field(default_factory=Limits)

    @classmethod
    def from_ismrmrd_encodingLimitsType(
        cls,
        encodingLimits: encodingLimitsType,
    ):
        values = {
            field.name: Limits.from_ismrmrd(getattr(encodingLimits, field.name))
            for field in dataclasses.fields(encodingLimits)
        }
        return cls(**values)
