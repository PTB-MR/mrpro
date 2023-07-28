"""Acceleration factor dataclass (DEPRECATED)."""

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

from dataclasses import dataclass

from ismrmrd.xsd.ismrmrdschema.ismrmrd import accelerationFactorType


@dataclass(slots=True)
class AccelerationFactor:
    """Acceleration Factor."""

    k1: float
    k2: float

    @property
    def overall(self) -> float:
        return self.k1 * self.k2

    @classmethod
    def from_ismrmrd(cls, data: accelerationFactorType) -> AccelerationFactor:
        """Create a AccelerationFactor from ismrmrd accelerationFactorType."""
        return cls(data.k1, data.k1)
