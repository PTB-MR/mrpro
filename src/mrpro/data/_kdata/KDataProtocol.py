"""Protocol for KData."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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

from typing import Literal, Protocol, Self

import torch
from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory


class _KDataProtocol(Protocol):
    """Protocol for KData used for type hinting in KData mixins.

    Note that the actual KData class can have more properties and methods than those defined here.

    If you want to use a property or method of KData in a new KDataMixin class,
    you must add it to this Protocol to make sure that the type hinting works [PRO]_.

    References
    ----------
    .. [PRO] Protocols https://typing.readthedocs.io/en/latest/spec/protocol.html#protocols
    """

    @property
    def header(self) -> KHeader: ...

    @property
    def data(self) -> torch.Tensor: ...

    @property
    def traj(self) -> KTrajectory: ...

    def __init__(self, header: KHeader, data: torch.Tensor, traj: KTrajectory): ...

    def _split_k2_or_k1_into_other(
        self,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
        split_dir: Literal['k1', 'k2'],
    ) -> Self: ...
