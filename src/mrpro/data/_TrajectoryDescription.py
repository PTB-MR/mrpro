"""TrajectoryDescription dataclass."""

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

from ismrmrd.xsd.ismrmrdschema.ismrmrd import trajectoryDescriptionType


@dataclass(slots=True)
class TrajectoryDescription:
    """TrajectoryDescription dataclass."""

    identifier: str = ''
    userParameterLong: dict[str, int] = dataclasses.field(default_factory=dict)
    userParameterDouble: dict[str, float] = dataclasses.field(default_factory=dict)
    userParameterString: dict[str, str] = dataclasses.field(default_factory=dict)
    comment: str = ''

    @classmethod
    def from_ismrmrd(cls, trajectoryDescription: trajectoryDescriptionType) -> TrajectoryDescription:
        """Create TrajectoryDescription from ismrmrd traj description."""

        return cls(
            userParameterLong={p.name: int(p.value) for p in trajectoryDescription.userParameterLong},
            userParameterDouble={p.name: float(p.value) for p in trajectoryDescription.userParameterDouble},
            userParameterString={p.name: str(p.value) for p in trajectoryDescription.userParameterString},
            comment=trajectoryDescription.comment or '',
            identifier=trajectoryDescription.identifier or '',
        )
