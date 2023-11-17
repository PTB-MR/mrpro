"""Tests of utilities for kdata objects."""

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
from einops import rearrange
from einops import repeat

from mrpro.data import AcqInfo
from mrpro.data import KData
from mrpro.data import KHeader
from mrpro.data import KTrajectory
from mrpro.utils import modify_acq_info
from mrpro.utils.kdata import combine_k2_k1_into_k1
from mrpro.utils.kdata import sel_other_subset_from_kdata
from mrpro.utils.kdata import split_idx
from mrpro.utils.kdata import split_k1_into_other
from mrpro.utils.kdata import split_k2_into_other
from tests.conftest import RandomGenerator
from tests.conftest import generate_random_data
from tests.conftest import generate_random_trajectory
from tests.conftest import random_acquisition
from tests.conftest import random_full_ismrmrd_header


@pytest.fixture(params=({'seed': 0, 'nother': 10, 'nk2': 40, 'nk1': 20},))
def random_kheader_shape(request, random_acquisition, random_full_ismrmrd_header):
    """Random (not necessarily valid) KHeader with defined shape."""
    # Get dimensions
    seed, nother, nk2, nk1 = (
        request.param['seed'],
        request.param['nother'],
        request.param['nk2'],
        request.param['nk1'],
    )
    generator = RandomGenerator(seed)

    # Generate acquisitions
    random_acq_info = AcqInfo.from_ismrmrd_acquisitions([random_acquisition for _ in range(nk1 * nk2 * nother)])
    nk0 = int(random_acq_info.number_of_samples[0])
    ncoils = int(random_acq_info.active_channels[0])

    # Generate trajectory
    ktraj = [generate_random_trajectory(generator, shape=(nk0, 2)) for _ in range(nk1 * nk2 * nother)]

    # Put it all together to a KHeader object
    kheader = KHeader.from_ismrmrd(random_full_ismrmrd_header, acq_info=random_acq_info, defaults={'trajectory': ktraj})
    return kheader, nother, ncoils, nk2, nk1, nk0


@pytest.fixture(params=({'seed': 0},))
def consistently_shaped_kdata(request, random_kheader_shape):
    """KData object with data, header and traj consistent in shape."""
    # Start with header
    kheader, nother, ncoils, nk2, nk1, nk0 = random_kheader_shape

    def reshape_acq_data(data):
        return rearrange(data, '(other k2 k1) ... -> other k2 k1 ...', other=nother, k2=nk2, k1=nk1)

    kheader.acq_info = modify_acq_info(reshape_acq_data, kheader.acq_info)

    # Create kdata with consistent shape
    kdat = generate_random_data(RandomGenerator(request.param['seed']), (nother, ncoils, nk2, nk1, nk0))

    # Create ktraj with consistent shape
    kx = repeat(torch.linspace(0, nk0 - 1, nk0, dtype=torch.float32), 'k0->other k2 k1 k0', other=1, k2=1, k1=1)
    ky = repeat(torch.linspace(0, nk1 - 1, nk1, dtype=torch.float32), 'k1->other k2 k1 k0', other=1, k2=1, k0=1)
    kz = repeat(torch.linspace(0, nk2 - 1, nk2, dtype=torch.float32), 'k2->other k2 k1 k0', other=1, k1=1, k0=1)
    ktraj = KTrajectory(kz, ky, kx)

    return KData(header=kheader, data=kdat, traj=ktraj)


def test_combine_k2_k1_into_k1(consistently_shaped_kdata):
    """Test the combindation of k2 and k1 dimension into k1."""
    # Create KData
    nother, ncoils, nk2, nk1, nk0 = consistently_shaped_kdata.data.shape

    # Combine data
    kdata_combined = combine_k2_k1_into_k1(consistently_shaped_kdata)

    # Verify shape of k-space data
    assert kdata_combined.data.shape == (nother, ncoils, 1, nk2 * nk1, nk0)
    # Verify shape of trajectory (it is the same for all other)
    assert kdata_combined.traj.broadcasted_shape == (1, 1, nk2 * nk1, nk0)


def test_modify_acq_info(random_kheader_shape):
    """Test the modification of the acquisition info."""
    # Create random header where AcqInfo fields are of shape [nk1*nk2] and reshape to [other, nk2, nk1]
    kheader, nother, _, nk2, nk1, _ = random_kheader_shape

    def reshape_acq_data(data):
        return rearrange(data, '(other k2 k1) ... -> other k2 k1 ...', other=nother, k2=nk2, k1=nk1)

    kheader.acq_info = modify_acq_info(reshape_acq_data, kheader.acq_info)

    # Verfiy shape
    assert kheader.acq_info.center_sample.shape == (nother, nk2, nk1)


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


@pytest.mark.parametrize(
    'nother_split,other_label',
    [
        (10, 'average'),
        (5, 'repetition'),
        (7, 'contrast'),
    ],
)
def test_split_k1_into_other(consistently_shaped_kdata, monkeypatch, nother_split, other_label):
    """Test splitting of the k1 dimension into other."""
    # Create KData
    nother, ncoils, nk2, nk1, nk0 = consistently_shaped_kdata.data.shape

    # Make sure that the other dimension/label used for the splitted data is not used yet
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'center', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'max', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'min', 0)

    # Create split index
    ni_per_block = nk1 // nother_split
    idx_k1 = torch.linspace(0, nk1 - 1, nk1, dtype=torch.int32)
    idx_split = split_idx(idx_k1, ni_per_block)

    # Split data
    kdata_split = split_k1_into_other(consistently_shaped_kdata, idx_split, other_label)

    # Verify shape of k-space data
    assert kdata_split.data.shape == (idx_split.shape[0] * nother, ncoils, nk2, ni_per_block, nk0)
    # Verify shape of trajectory
    assert kdata_split.traj.broadcasted_shape == (idx_split.shape[0] * nother, nk2, ni_per_block, nk0)
    # Verify new other label describes splitted data
    assert getattr(kdata_split.header.encoding_limits, other_label).length == idx_split.shape[0]


@pytest.mark.parametrize(
    'nother_split,other_label',
    [
        (10, 'average'),
        (5, 'repetition'),
        (7, 'contrast'),
    ],
)
def test_split_k2_into_other(consistently_shaped_kdata, monkeypatch, nother_split, other_label):
    """Test splitting of the k2 dimension into other."""
    # Create KData
    nother, ncoils, nk2, nk1, nk0 = consistently_shaped_kdata.data.shape

    # Make sure that the other dimension/label used for the splitted data is not used yet
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'center', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'max', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'min', 0)

    # Create split index
    ni_per_block = nk2 // nother_split
    idx_k2 = torch.linspace(0, nk2 - 1, nk2, dtype=torch.int32)
    idx_split = split_idx(idx_k2, ni_per_block)

    # Split data
    kdata_split = split_k2_into_other(consistently_shaped_kdata, idx_split, other_label)

    # Verify shape of k-space data
    assert kdata_split.data.shape == (idx_split.shape[0] * nother, ncoils, ni_per_block, nk1, nk0)
    # Verify shape of trajectory
    assert kdata_split.traj.broadcasted_shape == (idx_split.shape[0] * nother, ni_per_block, nk1, nk0)
    # Verify new other label describes splitted data
    assert getattr(kdata_split.header.encoding_limits, other_label).length == idx_split.shape[0]


@pytest.mark.parametrize(
    'subset_label,subset_idx',
    [
        ('repetition', torch.tensor([1], dtype=torch.int32)),
        ('average', torch.tensor([3, 4, 5], dtype=torch.int32)),
        ('phase', torch.tensor([2, 2, 8], dtype=torch.int32)),
    ],
)
def test_sel_other_subset_from_kdata(consistently_shaped_kdata, monkeypatch, subset_label, subset_idx):
    """Test selection of a subset from other dimension."""
    # Create KData
    nother, ncoil, nk2, nk1, nk0 = consistently_shaped_kdata.data.shape

    # Set required parameters used in sel_kdata_subset.
    _, iother, _ = torch.meshgrid(torch.arange(nk2), torch.arange(nother), torch.arange(nk1), indexing='xy')
    monkeypatch.setattr(consistently_shaped_kdata.header.acq_info.idx, subset_label, iother)

    # Select subset of data
    kdata_subset = sel_other_subset_from_kdata(consistently_shaped_kdata, subset_idx, subset_label)

    # Verify shape of data
    assert kdata_subset.data.shape == (subset_idx.shape[0], ncoil, nk2, nk1, nk0)
    # Verify other labe describes subset data
    assert all(torch.unique(getattr(kdata_subset.header.acq_info.idx, subset_label)) == torch.unique(subset_idx))
