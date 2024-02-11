"""Utilities for kdata objects."""

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

import torch
from einops import rearrange
from einops import repeat

from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data import Limits
from mrpro.utils import modify_acq_info


def combine_k2_k1_into_k1(
    kdata: KData,
) -> KData:
    """Reshape kdata from (... k2 k1 ...) to (... 1 (k2 k1) ...).

    Parameters
    ----------
    kdata
        K-space data (other coils k2 k1 k0)

    Returns
    -------
        K-space data (other coils 1 (k2 k1) k0)
    """

    # Rearrange data
    kdat = rearrange(kdata.data, 'other coils k2 k1 k0->other coils 1 (k2 k1) k0')

    # Rearrange trajectory
    ktraj = rearrange(kdata.traj.as_tensor(), 'dim other k2 k1 k0-> dim other 1 (k2 k1) k0')

    # Create new header with correct shape
    kheader = copy.deepcopy(kdata.header)

    # Update shape of acquisition info index
    def reshape_acq_info(info):
        return rearrange(info, 'other k2 k1 ... -> other 1 (k2 k1) ...')

    kheader.acq_info = modify_acq_info(reshape_acq_info, kheader.acq_info)

    return KData(kheader, kdat, KTrajectory.from_tensor(ktraj))


def split_idx(idx: torch.Tensor, np_per_block: int, np_overlap: int = 0, cyclic: bool = False) -> torch.Tensor:
    """Split a tensor of indices into different blocks.

    Parameters
    ----------
    idx
        1D indices to be split into different blocks.
    np_per_block
        Number of points per block.
    np_overlap, optional
        Number of points overlapping between blocks, by default 0, i.e. no overlap between blocks
    cyclic, optional
        Last block is filled up with points from the first block, e.g. due to cyclic cardiac motion, by default False


    Example:
    # idx = [1,2,3,4,5,6,7,8,9], np_per_block = 5, np_overlap = 3, cycle = True
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
    if np_overlap >= np_per_block:
        raise ValueError('Overlap has to be smaller than the number of points in a block.')

    # Calculate number of blocks
    # 1 2 3 4 5 6 7 8 9
    # x x                       step
    #     x x x                 np_overlap
    # x x x x x                 np_per_block
    step = np_per_block - np_overlap

    # For cyclic splitting utilize beginning of index to maximize number of blocks
    if cyclic:
        idx = torch.concat((idx, idx[:step]))

    return idx.unfold(dimension=0, size=np_per_block, step=step)


def split_k2_or_k1_into_other(
    kdata: KData,
    split_idx: torch.Tensor,
    other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    split_dir: Literal['k2', 'k1'],
) -> KData:
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
    if getattr(kdata.header.encoding_limits, other_label).length > 1:
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
    kdat = rearrange(split_data_traj(kdata.data), rearrange_pattern_data)

    # First we need to make sure the other dimension is the same as data then we can split the trajectory
    ktraj = kdata.traj.as_tensor()
    # Verify that other dimension of trajectory is 1 or matches data
    if ktraj.shape[1] > 1 and ktraj.shape[1] != kdata.data.shape[0]:
        raise ValueError(f'other dimension of trajectory has to be 1 or match data ({kdata.data.shape[0]})')
    elif ktraj.shape[1] == 1 and kdata.data.shape[0] > 1:
        ktraj = repeat(ktraj, 'dim other k2 k1 k0->dim (other_data other) k2 k1 k0', other_data=kdata.data.shape[0])
    ktraj = rearrange(split_data_traj(ktraj), rearrange_pattern_traj)

    # Create new header with correct shape
    kheader = copy.deepcopy(kdata.header)

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

    return KData(kheader, kdat, KTrajectory.from_tensor(ktraj))


def split_k1_into_other(
    kdata: KData,
    split_idx: torch.Tensor,
    other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
) -> KData:
    """Based on an index tensor, split the data in e.g. phases.

    Parameters
    ----------
    kdata
        K-space data (other coils k2 k1 k0)
    split_idx
        2D index describing the k1 points in each block to be moved to the other dimension  (other_split, k1_per_split)
    other_label
        Label of other dimension, e.g. repetition, phase

    Returns
    -------
        K-space data with new shape ((other other_split) coils k2 k1_per_split k0)
    """
    return split_k2_or_k1_into_other(kdata, split_idx, other_label, split_dir='k1')


def split_k2_into_other(
    kdata: KData,
    split_idx: torch.Tensor,
    other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
) -> KData:
    """Based on an index tensor, split the data in e.g. phases.

    Parameters
    ----------
    kdata
        K-space data (other coils k2 k1 k0)
    split_idx
        2D index describing the k2 points in each block to be moved to the other dimension  (other_split, k2_per_split)
    other_label
        Label of other dimension, e.g. repetition, phase

    Returns
    -------
        K-space data with new shape ((other other_split) coils k2_per_split k1 k0)
    """
    return split_k2_or_k1_into_other(kdata, split_idx, other_label, split_dir='k2')


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
