"""Modify the shape of data objects."""

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

import copy
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from einops import repeat

from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data import Limits
from mrpro.utils import modify_acq_info


def split_idx(idx: torch.Tensor, ni_per_block: int, ni_overlap: int = 0, cyclic: bool = False) -> torch.Tensor:
    """Split a tensor of indices into different blocks.

    Parameters
    ----------
    idx
        1D indices to be split into different blocks.
    ni_per_block
        Number of points per block.
    ni_overlap
        Number of points overlapping between blocks, by default 0, i.e. no overlap between blocks
    cyclic, optional
        Last block is filled up with points from the first block, e.g. due to cyclic cardiac motion, by default False


    Example:
    # idx = [1,2,3,4,5,6,7,8,9], ni_per_block = 5, ni_overlap = 3, cycle = True
    #
    # idx:     1 2 3 4 5 6 7 8 9
    # block 0: 0 0 0 0 0
    # block 1:     1 1 1 1 1
    # block 2:         2 2 2 2 2
    # block 3: 3 3         3 3 3

    Returns
    -------
        2D indices to split data into different blocks in the shape [block, index].

    Raises
    ------
    ValueError
        If the provided idx is not 1D
    ValueError
        If the overlap is smaller than the number of points per block
    """

    # Make sure idx is 1D
    if idx.ndim != 1:
        raise ValueError('idx should be a 1D vector.')

    # Make sure overlap is not larger than the number of points in a block
    if ni_overlap >= ni_per_block:
        raise ValueError('Overlap has to be smaller than the number of points in a block.')

    # Calculate number of blocks
    # 1 2 3 4 5 6 7 8 9
    # x x                       ni_non_overlap
    #     x x x                 ni_overlap
    # x x x x x                 ni_per_block
    ni_non_overlap = ni_per_block - ni_overlap

    # For cyclic splitting utilize beginning of index to maximize number of blocks
    if cyclic:
        num_blocks = int(np.ceil(idx.shape[0] / ni_non_overlap))
        # Add required number of points from the beginning to the end
        idx = torch.cat((idx, idx[: (num_blocks * ni_non_overlap + ni_overlap) - idx.shape[0]]), dim=0)
    else:
        num_blocks = (idx.shape[0] - ni_overlap) // ni_non_overlap

    # Go through each block and get indices
    split_idx = torch.zeros((num_blocks, ni_per_block), dtype=idx.dtype)
    for block_idx in range(num_blocks):
        block_start = block_idx * ni_non_overlap
        block_end = block_start + ni_per_block
        split_idx[block_idx, :] = idx[block_start:block_end]

    return split_idx


def split_k1_into_other(
    kdata: KData,
    split_idx: torch.Tensor,
    other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
) -> KData:
    """Based on an index tensor, split the data in e.g. phases.

    Parameters
    ----------
    kdata
        K-space data.
    split_idx
        2D index describing the k1 points in each block to be moved to the other dimension.
    other_label
        Label of other dimension, e.g. repetition, phase

    Returns
    -------
        K-space data with new shape.

    Raises
    ------
    ValueError
        Already exisiting "other_label" can only be of length 1
    """

    # Number of other
    num_other = split_idx.shape[0]

    # Verify that the specified label of the other dimension is unused
    if getattr(kdata.header.encoding_limits, other_label).length > 1:
        raise ValueError(f'{other_label} is already used to encode different parts of the scan.')

    # Split data
    kdat = rearrange(
        kdata.data[:, :, :, split_idx, :],
        'other coils k2 other_split k1 k0->(other other_split) coils k2 k1 k0',
        other_split=num_other,
    )

    # Split trajectory
    ktraj = rearrange(
        kdata.traj.as_tensor()[:, :, :, split_idx, :],
        'dim other k2 other_split k1 k0->dim (other other_split) k2 k1 k0',
        other_split=num_other,
    )

    # Create new header with correct shape
    kheader = copy.deepcopy(kdata.header)

    # Update shape of acquisition info index
    def reshape_acq_info(info):
        return rearrange(
            info[:, :, split_idx, ...],
            'other k2 other_split k1 ... -> (other other_split) k2 k1 ...',
            other_split=num_other,
        )

    kheader.acq_info = modify_acq_info(reshape_acq_info, kheader.acq_info)

    # Update other label limits and acquisition info
    setattr(kheader.encoding_limits, other_label, Limits(min=0, max=num_other - 1, center=0))
    setattr(
        kheader.acq_info.idx,
        other_label,
        repeat(torch.linspace(0, num_other - 1, num_other), 'other-> other 1 k1', k1=split_idx.shape[1]),
    )

    return KData(kheader, kdat, KTrajectory.from_tensor(ktraj))


def sel_other_subset_from_kdata(
    kdata: KData,
    subset_idx: torch.Tensor,
    subset_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
) -> KData:
    # Make a copy such that the original kdata.header remains the same
    kheader = copy.deepcopy(kdata.header)
    ktraj = kdata.traj.as_tensor()

    # Verify that the subset_idx is available
    label_idx = getattr(kheader.acq_info.idx, subset_label)
    if not all([el in torch.unique(label_idx) for el in subset_idx]):
        raise ValueError('Subset indices are outside of the available index range')

    # Find subset index in acq_info index
    other_idx = torch.cat([torch.where(idx == label_idx[:, 0, 0])[0] for idx in subset_idx], dim=0)

    # Adapt header
    def select_acq_info(info):
        return info[other_idx, ...]

    kheader.acq_info = modify_acq_info(select_acq_info, kheader.acq_info)

    # Select data
    kdat = kdata.data[other_idx, ...]

    # Select ktraj
    if ktraj.shape[1] > 1:
        ktraj = ktraj[:, other_idx, ...]

    return KData(kheader, kdat, KTrajectory.from_tensor(ktraj))
