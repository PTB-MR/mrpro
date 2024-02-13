"""Tests of split index calculation."""

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

import pytest
import torch
from einops import repeat

from mrpro.utils import split_idx


@pytest.mark.parametrize(
    'ni_per_block,ni_overlap,cyclic,unique_values_in_last_block',
    [
        (5, 0, False, torch.tensor([3])),
        (6, 2, False, torch.tensor([2, 3])),
        (6, 2, True, torch.tensor([0, 3])),
    ],
)
def test_split_idx(ni_per_block, ni_overlap, cyclic, unique_values_in_last_block):
    """Test the calculation of indices to split data into different blocks."""
    # Create a regular sequence of values
    vals = repeat(torch.tensor([0, 1, 2, 3]), 'd0 -> (d0 repeat)', repeat=5)
    # Mix up values
    vals = vals[torch.randperm(vals.shape[0])]

    # Split indices of sorted sequence
    idx_split = split_idx(torch.argsort(vals), ni_per_block, ni_overlap, cyclic)

    # Split sequence of values
    vals = vals[idx_split]

    # Make sure sequence is correctly split
    torch.testing.assert_close(torch.unique(vals[-1, :]), unique_values_in_last_block)
