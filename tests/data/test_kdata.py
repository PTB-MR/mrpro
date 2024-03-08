"""Tests for the KData class."""

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
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators._KTrajectoryCalculator import DummyTrajectory
from mrpro.operators import FastFourierOp
from mrpro.utils import modify_acq_info
from mrpro.utils import split_idx

from tests.conftest import RandomGenerator
from tests.conftest import generate_random_data
from tests.data import IsmrmrdRawTestData
from tests.helper import relative_image_difference


@pytest.fixture(scope='session')
def ismrmrd_cart(ellipse_phantom, tmp_path_factory):
    """Fully sampled cartesian data set."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        phantom=ellipse_phantom.phantom,
    )
    return ismrmrd_kdata


@pytest.fixture(scope='session')
def ismrmrd_cart_invalid_reps(tmp_path_factory):
    """Fully sampled cartesian data set."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        flag_invalid_reps=True,
    )
    return ismrmrd_kdata


@pytest.fixture(scope='session')
def ismrmrd_cart_random_us(ellipse_phantom, tmp_path_factory):
    """Randomly undersampled cartesian data set with repetitions."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        acceleration=4,
        sampling_order='random',
        phantom=ellipse_phantom.phantom,
    )
    return ismrmrd_kdata


@pytest.fixture(params=({'seed': 0},))
def consistently_shaped_kdata(request, random_kheader_shape):
    """KData object with data, header and traj consistent in shape."""
    # Start with header
    kheader, n_other, n_coils, n_k2, n_k1, n_k0 = random_kheader_shape

    def reshape_acq_data(data):
        return rearrange(data, '(other k2 k1) ... -> other k2 k1 ...', other=n_other, k2=n_k2, k1=n_k1)

    kheader.acq_info = modify_acq_info(reshape_acq_data, kheader.acq_info)

    # Create kdata with consistent shape
    kdata = generate_random_data(RandomGenerator(request.param['seed']), (n_other, n_coils, n_k2, n_k1, n_k0))

    # Create ktraj with consistent shape
    kx = repeat(torch.linspace(0, n_k0 - 1, n_k0, dtype=torch.float32), 'k0->other k2 k1 k0', other=1, k2=1, k1=1)
    ky = repeat(torch.linspace(0, n_k1 - 1, n_k1, dtype=torch.float32), 'k1->other k2 k1 k0', other=1, k2=1, k0=1)
    kz = repeat(torch.linspace(0, n_k2 - 1, n_k2, dtype=torch.float32), 'k2->other k2 k1 k0', other=1, k1=1, k0=1)
    ktraj = KTrajectory(kz, ky, kx)

    return KData(header=kheader, data=kdata, traj=ktraj)


def test_KData_from_file(ismrmrd_cart):
    """Read in data from file."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert kdata is not None


def test_KData_random_cart_undersampling(ismrmrd_cart_random_us):
    """Read data with different random Cartesian undersampling in multiple
    repetitions."""
    kdata = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    assert kdata is not None


def test_KData_random_cart_undersampling_shape(ismrmrd_cart_random_us):
    """Check shape of KData with random Cartesian undersampling."""
    kdata = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    # check if the number of repetitions is correct
    assert kdata.data.shape[-5] == ismrmrd_cart_random_us.repetitions
    # check if the number of phase encoding lines per repetition is correct
    assert kdata.data.shape[-2] == ismrmrd_cart_random_us.matrix_size // ismrmrd_cart_random_us.acceleration


def test_KData_raise_wrong_trajectory_shape(ismrmrd_cart):
    """Wrong KTrajectory shape raises exception."""
    kx = ky = kz = torch.zeros(1, 2, 3, 4)
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=None)
    with pytest.raises(ValueError):
        _ = KData.from_file(ismrmrd_cart.filename, trajectory)


def test_KData_from_file_diff_nky_for_rep(ismrmrd_cart_invalid_reps):
    """Multiple repetitions with different number of phase encoding lines."""
    with pytest.warns(UserWarning, match=r'different number'):
        kdata = KData.from_file(ismrmrd_cart_invalid_reps.filename, DummyTrajectory())
    assert kdata.data.shape[-2] == 1, 'k1 should be 1'
    assert kdata.data.shape[-3] == 1, 'k2 should be 1'


def test_KData_kspace(ismrmrd_cart):
    """Read in data and verify k-space by comparing reconstructed image."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    ff_op = FastFourierOp(dim=(-1, -2))
    (reconstructed_img,) = ff_op.adjoint(kdata.data)

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img[0, 0, 0, ...], ismrmrd_cart.img_ref) <= 0.05


@pytest.mark.parametrize(('field', 'value'), [('b0', 11.3), ('tr', [24.3])])
def test_KData_modify_header(ismrmrd_cart, field, value):
    """Overwrite some parameters in the header."""
    parameter_dict = {field: value}
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory(), header_overwrites=parameter_dict)
    assert getattr(kdata.header, field) == value


def test_KData_to_complex128(ismrmrd_cart):
    """Change KData dtype complex128."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata_complex128.data.dtype == torch.complex128
    assert kdata_complex128.traj.kx.dtype == torch.float64
    assert kdata_complex128.traj.ky.dtype == torch.float64
    assert kdata_complex128.traj.kz.dtype == torch.float64


@pytest.mark.cuda()
def test_KData_to_cuda(ismrmrd_cart):
    """Test KData.to to move data to CUDA memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    cuda_device = f'cuda:{torch.cuda.current_device()}'
    kdata_cuda = kdata.to(device=cuda_device)
    assert kdata_cuda.data.is_cuda
    assert kdata_cuda.traj.kz.is_cuda
    assert kdata_cuda.traj.ky.is_cuda
    assert kdata_cuda.traj.kx.is_cuda


@pytest.mark.cuda()
def test_KData_cuda(ismrmrd_cart):
    """Move KData object to CUDA memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cuda = kdata.cuda()
    assert kdata_cuda.data.is_cuda
    assert kdata_cuda.traj.kz.is_cuda
    assert kdata_cuda.traj.ky.is_cuda
    assert kdata_cuda.traj.kx.is_cuda


@pytest.mark.cuda()
def test_KData_cpu(ismrmrd_cart):
    """Move KData object to CUDA memory and back to CPU memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cpu = kdata.cuda().cpu()
    assert kdata_cpu.data.is_cpu
    assert kdata_cpu.traj.kz.is_cpu
    assert kdata_cpu.traj.ky.is_cpu
    assert kdata_cpu.traj.kx.is_cpu


def test_KData_rearrange_k2_k1_into_k1(consistently_shaped_kdata):
    """Test rearranging of k2 and k1 dimension into k1."""
    # Create KData
    n_other, n_coils, n_k2, n_k1, n_k0 = consistently_shaped_kdata.data.shape

    # Combine data
    kdata_combined = consistently_shaped_kdata.rearrange_k2_k1_into_k1()

    # Verify shape of k-space data
    assert kdata_combined.data.shape == (n_other, n_coils, 1, n_k2 * n_k1, n_k0)
    # Verify shape of trajectory (it is the same for all other)
    assert kdata_combined.traj.broadcasted_shape == (1, 1, n_k2 * n_k1, n_k0)


@pytest.mark.parametrize(
    ('n_other_split', 'other_label'),
    [
        (10, 'average'),
        (5, 'repetition'),
        (7, 'contrast'),
    ],
)
def test_KData_split_k1_into_other(consistently_shaped_kdata, monkeypatch, n_other_split, other_label):
    """Test splitting of the k1 dimension into other."""
    # Create KData
    n_other, n_coils, n_k2, n_k1, n_k0 = consistently_shaped_kdata.data.shape

    # Make sure that the other dimension/label used for the split data is not used yet
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'center', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'max', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'min', 0)

    # Create split index
    ni_per_block = n_k1 // n_other_split
    idx_k1 = torch.linspace(0, n_k1 - 1, n_k1, dtype=torch.int32)
    idx_split = split_idx(idx_k1, ni_per_block)

    # Split data
    kdata_split = consistently_shaped_kdata.split_k1_into_other(idx_split, other_label)

    # Verify shape of k-space data
    assert kdata_split.data.shape == (idx_split.shape[0] * n_other, n_coils, n_k2, ni_per_block, n_k0)
    # Verify shape of trajectory
    assert kdata_split.traj.broadcasted_shape == (idx_split.shape[0] * n_other, n_k2, ni_per_block, n_k0)
    # Verify new other label describes split data
    assert getattr(kdata_split.header.encoding_limits, other_label).length == idx_split.shape[0]


@pytest.mark.parametrize(
    ('n_other_split', 'other_label'),
    [
        (10, 'average'),
        (5, 'repetition'),
        (7, 'contrast'),
    ],
)
def test_KData_split_k2_into_other(consistently_shaped_kdata, monkeypatch, n_other_split, other_label):
    """Test splitting of the k2 dimension into other."""
    # Create KData
    n_other, n_coils, n_k2, n_k1, n_k0 = consistently_shaped_kdata.data.shape

    # Make sure that the other dimension/label used for the split data is not used yet
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'center', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'max', 0)
    monkeypatch.setattr(getattr(consistently_shaped_kdata.header.encoding_limits, other_label), 'min', 0)

    # Create split index
    ni_per_block = n_k2 // n_other_split
    idx_k2 = torch.linspace(0, n_k2 - 1, n_k2, dtype=torch.int32)
    idx_split = split_idx(idx_k2, ni_per_block)

    # Split data
    kdata_split = consistently_shaped_kdata.split_k2_into_other(idx_split, other_label)

    # Verify shape of k-space data
    assert kdata_split.data.shape == (idx_split.shape[0] * n_other, n_coils, ni_per_block, n_k1, n_k0)
    # Verify shape of trajectory
    assert kdata_split.traj.broadcasted_shape == (idx_split.shape[0] * n_other, ni_per_block, n_k1, n_k0)
    # Verify new other label describes split data
    assert getattr(kdata_split.header.encoding_limits, other_label).length == idx_split.shape[0]


@pytest.mark.parametrize(
    ('subset_label', 'subset_idx'),
    [
        ('repetition', torch.tensor([1], dtype=torch.int32)),
        ('average', torch.tensor([3, 4, 5], dtype=torch.int32)),
        ('phase', torch.tensor([2, 2, 8], dtype=torch.int32)),
    ],
)
def test_KData_select_other_subset(consistently_shaped_kdata, monkeypatch, subset_label, subset_idx):
    """Test selection of a subset from other dimension."""
    # Create KData
    n_other, n_coils, n_k2, n_k1, n_k0 = consistently_shaped_kdata.data.shape

    # Set required parameters used in sel_kdata_subset.
    _, iother, _ = torch.meshgrid(torch.arange(n_k2), torch.arange(n_other), torch.arange(n_k1), indexing='xy')
    monkeypatch.setattr(consistently_shaped_kdata.header.acq_info.idx, subset_label, iother)

    # Select subset of data
    kdata_subset = consistently_shaped_kdata.select_other_subset(subset_idx, subset_label)

    # Verify shape of data
    assert kdata_subset.data.shape == (subset_idx.shape[0], n_coils, n_k2, n_k1, n_k0)
    # Verify other label describes subset data
    assert all(torch.unique(getattr(kdata_subset.header.acq_info.idx, subset_label)) == torch.unique(subset_idx))
