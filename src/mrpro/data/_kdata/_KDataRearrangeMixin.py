"""Rearrange KData."""

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

import copy
from typing import TYPE_CHECKING

from einops import rearrange

if TYPE_CHECKING:
    from mrpro.data._kdata._KData import _KDataProtocol

from mrpro.data._AcqInfo import AcqInfo
from mrpro.utils import modify_acq_info


class KDataRearrangeMixin:
    """Rearrange KData."""

    def rearrange_k2_k1_into_k1(self: _KDataProtocol) -> _KDataProtocol:
        """Rearrange kdata from (... k2 k1 ...) to (... 1 (k2 k1) ...).

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)

        Returns
        -------
            K-space data (other coils 1 (k2 k1) k0)
        """
        # Rearrange data
        kdat = rearrange(self.data, '... coils k2 k1 k0->... coils 1 (k2 k1) k0')

        # Rearrange trajectory
        ktraj = rearrange(self.traj.as_tensor(), 'dim ... k2 k1 k0-> dim ... 1 (k2 k1) k0')

        # Create new header with correct shape
        kheader = copy.deepcopy(self.header)

        # Update shape of acquisition info index
        def reshape_acq_info(info: AcqInfo):
            return rearrange(info, 'other k2 k1 ... -> other 1 (k2 k1) ...')

        kheader.acq_info = modify_acq_info(reshape_acq_info, kheader.acq_info)

        return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))
