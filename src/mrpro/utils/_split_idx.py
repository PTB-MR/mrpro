"""Index for splitting data."""

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

import torch


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
