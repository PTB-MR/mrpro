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

from dataclasses import astuple
from dataclasses import dataclass
from dataclasses import fields

from ismrmrd.xsd.ismrmrdschema.ismrmrd import encodingLimitsType
from ismrmrd.xsd.ismrmrdschema.ismrmrd import limitType


@dataclass(slots=True)
class Limits:
    """Limits dataclass with min, max, and center attributes."""

    min: int
    max: int
    center: int

    @classmethod
    def from_ismrmrd_limitType(cls, limitType: limitType):
        if limitType is None:
            return cls(0, 0, 0)
        return cls(*astuple(limitType))


@dataclass(slots=True)
class EncodingLimits:
    """Encoding limits dataclass with limits for each attribute."""

    kspace_encoding_step_0: Limits
    kspace_encoding_step_1: Limits
    kspace_encoding_step_2: Limits
    average: Limits
    slice: Limits
    contrast: Limits
    phase: Limits
    repetition: Limits
    set: Limits
    segment: Limits

    @classmethod
    def from_ismrmrd_encodingLimitsType(
        cls,
        encodingLimitsType: encodingLimitsType,
    ):
        values = {
            field.name: Limits.from_ismrmrd_limitType(
                getattr(encodingLimitsType, field.name)
            )
            for field in fields(cls)
        }
        return cls(**values)
