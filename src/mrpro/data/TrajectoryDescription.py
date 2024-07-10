"""TrajectoryDescription dataclass."""

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

import dataclasses
from dataclasses import dataclass
from typing import Self

from ismrmrd.xsd.ismrmrdschema.ismrmrd import trajectoryDescriptionType


@dataclass(slots=True)
class TrajectoryDescription:
    """TrajectoryDescription dataclass."""

    identifier: str = ''
    user_parameter_long: dict[str, int] = dataclasses.field(default_factory=dict)
    user_parameter_double: dict[str, float] = dataclasses.field(default_factory=dict)
    user_parameter_string: dict[str, str] = dataclasses.field(default_factory=dict)
    comment: str = ''

    @classmethod
    def from_ismrmrd(cls, trajectory_description: trajectoryDescriptionType) -> Self:
        """Create TrajectoryDescription from ismrmrd traj description."""
        return cls(
            user_parameter_long={p.name: int(p.value) for p in trajectory_description.userParameterLong},
            user_parameter_double={p.name: float(p.value) for p in trajectory_description.userParameterDouble},
            user_parameter_string={p.name: str(p.value) for p in trajectory_description.userParameterString},
            comment=trajectory_description.comment or '',
            identifier=trajectory_description.identifier or '',
        )
