"""Tests of split index calculation."""

import pytest
import torch
from einops import repeat
from mrpro.utils import split_idx

from tests import RandomGenerator


@pytest.mark.parametrize(
    ('ni_per_block', 'ni_overlap', 'cyclic', 'unique_values_in_last_block'),
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
    vals = vals[RandomGenerator(13).randperm(vals.shape[0])]

    # Split indices of sorted sequence
    idx_split = split_idx(torch.argsort(vals), ni_per_block, ni_overlap, cyclic)

    # Split sequence of values
    vals = vals[idx_split]

    # Make sure sequence is correctly split
    torch.testing.assert_close(torch.unique(vals[-1, :]), unique_values_in_last_block)
