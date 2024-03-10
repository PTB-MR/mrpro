"""Mixin class to split KData into other subsets."""

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
from typing import Literal

import torch
from einops import rearrange
from einops import repeat

if TYPE_CHECKING:
    from mrpro.data._kdata._KData import _KDataProtocol

from mrpro.data import Limits
from mrpro.utils import modify_acq_info


class KDataSplitMixin:
    """Split KData into other subsets."""

    def _split_k2_or_k1_into_other(
        self: _KDataProtocol,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
        split_dir: Literal['k2', 'k1'],
    ) -> _KDataProtocol:
        """Based on an index tensor, split the data in e.g. phases.

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)
        split_idx
            2D index describing the k2 or k1 points in each block to be moved to the other dimension
            (other_split, k1_per_split) or (other_split, k2_per_split)
        other_label
            Label of other dimension, e.g. repetition, phase

        Returns
        -------
            K-space data with new shape
            ((other other_split) coils k2 k1_per_split k0) or ((other other_split) coils k2_per_split k1 k0)

        Raises
        ------
        ValueError
            Already exisiting "other_label" can only be of length 1
        """

        # Number of other
        num_other = split_idx.shape[0]

        # Verify that the specified label of the other dimension is unused
        if getattr(self.header.encoding_limits, other_label).length > 1:
            raise ValueError(f'{other_label} is already used to encode different parts of the scan.')

        # Set-up splitting
        if split_dir == 'k1':
            # Split along k1 dimensions
            def split_data_traj(dat_traj):
                return dat_traj[:, :, :, split_idx, :]

            def split_acq_info(acq_info):
                return acq_info[:, :, split_idx, ...]

            # Rearrange other_split and k1 dimension
            rearrange_pattern_data = 'other coils k2 other_split k1 k0->(other other_split) coils k2 k1 k0'
            rearrange_pattern_traj = 'dim other k2 other_split k1 k0->dim (other other_split) k2 k1 k0'
            rearrange_pattern_acq_info = 'other k2 other_split k1 ... -> (other other_split) k2 k1 ...'

        elif split_dir == 'k2':
            # Split along k2 dimensions
            def split_data_traj(dat_traj):
                return dat_traj[:, :, split_idx, :, :]

            def split_acq_info(acq_info):
                return acq_info[:, split_idx, ...]

            # Rearrange other_split and k1 dimension
            rearrange_pattern_data = 'other coils other_split k2 k1 k0->(other other_split) coils k2 k1 k0'
            rearrange_pattern_traj = 'dim other other_split k2 k1 k0->dim (other other_split) k2 k1 k0'
            rearrange_pattern_acq_info = 'other other_split k2 k1 ... -> (other other_split) k2 k1 ...'

        else:
            raise ValueError('split_dir has to be "k1" or "k2"')

        # Split data
        kdat = rearrange(split_data_traj(self.data), rearrange_pattern_data)

        # First we need to make sure the other dimension is the same as data then we can split the trajectory
        ktraj = self.traj.as_tensor()
        # Verify that other dimension of trajectory is 1 or matches data
        if ktraj.shape[1] > 1 and ktraj.shape[1] != self.data.shape[0]:
            raise ValueError(f'other dimension of trajectory has to be 1 or match data ({self.data.shape[0]})')
        elif ktraj.shape[1] == 1 and self.data.shape[0] > 1:
            ktraj = repeat(ktraj, 'dim other k2 k1 k0->dim (other_data other) k2 k1 k0', other_data=self.data.shape[0])
        ktraj = rearrange(split_data_traj(ktraj), rearrange_pattern_traj)

        # Create new header with correct shape
        kheader = copy.deepcopy(self.header)

        # Update shape of acquisition info index
        def reshape_acq_info(info):
            return rearrange(split_acq_info(info), rearrange_pattern_acq_info)

        kheader.acq_info = modify_acq_info(reshape_acq_info, kheader.acq_info)

        # Update other label limits and acquisition info
        setattr(kheader.encoding_limits, other_label, Limits(min=0, max=num_other - 1, center=0))

        # acq_info for new other dimensions
        acq_info_other_split = repeat(
            torch.linspace(0, num_other - 1, num_other), 'other-> other k2 k1', k2=kdat.shape[-3], k1=kdat.shape[-2]
        )
        setattr(kheader.acq_info.idx, other_label, acq_info_other_split)

        return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))

    def split_k1_into_other(
        self: _KDataProtocol,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> _KDataProtocol:
        """Based on an index tensor, split the data in e.g. phases.

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)
        split_idx
            2D index describing the k1 points in each block to be moved to other dimension  (other_split, k1_per_split)
        other_label
            Label of other dimension, e.g. repetition, phase

        Returns
        -------
            K-space data with new shape ((other other_split) coils k2 k1_per_split k0)
        """
        return self._split_k2_or_k1_into_other(split_idx, other_label, split_dir='k1')

    def split_k2_into_other(
        self: _KDataProtocol,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> _KDataProtocol:
        """Based on an index tensor, split the data in e.g. phases.

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)
        split_idx
            2D index describing the k2 points in each block to be moved to other dimension  (other_split, k2_per_split)
        other_label
            Label of other dimension, e.g. repetition, phase

        Returns
        -------
            K-space data with new shape ((other other_split) coils k2_per_split k1 k0)
        """
        return self._split_k2_or_k1_into_other(split_idx, other_label, split_dir='k2')
