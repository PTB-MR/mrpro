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

from mrpro.data import AcqInfo
from mrpro.data import KData
from mrpro.data import KHeader
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators._KTrajectoryCalculator import DummyTrajectory
from mrpro.operators import FastFourierOp
from mrpro.utils import modify_acq_info
from mrpro.utils import split_idx
from tests.conftest import RandomGenerator
from tests.conftest import generate_random_data
from tests.conftest import generate_random_trajectory
from tests.conftest import random_acquisition
from tests.conftest import random_full_ismrmrd_header
from tests.data import IsmrmrdRawTestData
from tests.helper import rel_image_diff
from tests.phantoms.test_ellipse_phantom import ph_ellipse


@pytest.fixture(scope='session')
def ismrmrd_cart(ph_ellipse, tmp_path_factory):
    """Fully sampled cartesian data set."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawTestData(
        filename=ismrmrd_filename, noise_level=0.0, repetitions=3, phantom=ph_ellipse.phantom
    )
    return ismrmrd_kdat


@pytest.fixture(scope='session')
def ismrmrd_cart_invalid_reps(tmp_path_factory):
    """Fully sampled cartesian data set."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawTestData(filename=ismrmrd_filename, noise_level=0.0, repetitions=3, flag_invalid_reps=True)
    return ismrmrd_kdat


@pytest.fixture(scope='session')
def ismrmrd_cart_random_us(ph_ellipse, tmp_path_factory):
    """Randomly undersampled cartesian data set with repetitions."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        acceleration=4,
        sampling_order='random',
        phantom=ph_ellipse.phantom,
    )
    return ismrmrd_kdat


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


def test_KData_from_file(ismrmrd_cart):
    """Read in data from file."""
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert k is not None


def test_KData_random_cart_undersampling(ismrmrd_cart_random_us):
    """Read data with different random Cartesian undersampling in multiple
    repetitions."""
    k = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    assert k is not None


def test_KData_random_cart_undersampling_shape(ismrmrd_cart_random_us):
    """Check shape of KData with random Cartesian undersampling."""
    k = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    # check if the number of repetitions is correct
    assert k.data.shape[-5] == ismrmrd_cart_random_us.repetitions
    # check if the number of phase encoding lines per repetition is correct
    assert k.data.shape[-2] == ismrmrd_cart_random_us.matrix_size // ismrmrd_cart_random_us.acceleration


def test_KData_raise_wrong_ktraj_shape(ismrmrd_cart):
    """Wrong KTrajectory shape raises exception."""
    kx = ky = kz = torch.zeros(1, 2, 3, 4)
    ktraj = KTrajectory(kz, ky, kx, repeat_detection_tolerance=None)
    with pytest.raises(ValueError):
        _ = KData.from_file(ismrmrd_cart.filename, ktraj)


def test_KData_from_file_diff_nky_for_rep(ismrmrd_cart_invalid_reps):
    """Multiple repetitions with different number of phase encoding lines is
    not supported."""
    with pytest.raises(ValueError, match=r'Number of \((k2 k1\)) points in '):
        KData.from_file(ismrmrd_cart_invalid_reps.filename, DummyTrajectory())


def test_KData_kspace(ismrmrd_cart):
    """Read in data and verify k-space by comparing reconstructed image."""
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    FFOp = FastFourierOp(dim=(-1, -2))
    irec = FFOp.adjoint(k.data)

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert rel_image_diff(irec[0, 0, 0, ...], ismrmrd_cart.imref) <= 0.05


@pytest.mark.parametrize('field,value', [('b0', 11.3), ('tr', [24.3])])
def test_KData_modify_header(ismrmrd_cart, field, value):
    """Overwrite some parameters in the header."""
    par_dict = {field: value}
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory(), header_overwrites=par_dict)
    assert getattr(k.header, field) == value


def test_KData_to_complex128(ismrmrd_cart):
    """Change KData dtype complex128."""
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    k_complex128 = k.to(dtype=torch.complex128)
    assert k_complex128.data.dtype == torch.complex128
    assert k_complex128.traj.kx.dtype == torch.float64
    assert k_complex128.traj.ky.dtype == torch.float64
    assert k_complex128.traj.kz.dtype == torch.float64


@pytest.mark.cuda
def test_KData_to_cuda(ismrmrd_cart):
    """Test KData.to to move data to CUDA memory."""
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    cuda_device = f'cuda:{torch.cuda.current_device()}'
    kcuda = k.to(device=cuda_device)
    assert kcuda.data.is_cuda
    assert kcuda.traj.kz.is_cuda
    assert kcuda.traj.ky.is_cuda
    assert kcuda.traj.kx.is_cuda


@pytest.mark.cuda
def test_KData_cuda(ismrmrd_cart):
    """Move KData object to CUDA memory."""
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kcuda = k.cuda()
    assert kcuda.data.is_cuda
    assert kcuda.traj.kz.is_cuda
    assert kcuda.traj.ky.is_cuda
    assert kcuda.traj.kx.is_cuda


@pytest.mark.cuda
def test_KData_cpu(ismrmrd_cart):
    """Move KData object to CUDA memory and back to CPU memory."""
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kcpu = k.cuda().cpu()
    assert kcpu.data.is_cpu
    assert kcpu.traj.kz.is_cpu
    assert kcpu.traj.ky.is_cpu
    assert kcpu.traj.kx.is_cpu


def test_KData_rearrange_k2_k1_into_k1(consistently_shaped_kdata):
    """Test rearranging of k2 and k1 dimension into k1."""
    # Create KData
    nother, ncoils, nk2, nk1, nk0 = consistently_shaped_kdata.data.shape

    # Combine data
    kdata_combined = consistently_shaped_kdata.rearrange_k2_k1_into_k1()

    # Verify shape of k-space data
    assert kdata_combined.data.shape == (nother, ncoils, 1, nk2 * nk1, nk0)
    # Verify shape of trajectory (it is the same for all other)
    assert kdata_combined.traj.broadcasted_shape == (1, 1, nk2 * nk1, nk0)


@pytest.mark.parametrize(
    'nother_split,other_label',
    [
        (10, 'average'),
        (5, 'repetition'),
        (7, 'contrast'),
    ],
)
def test_KData_split_k1_into_other(consistently_shaped_kdata, monkeypatch, nother_split, other_label):
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
    kdata_split = consistently_shaped_kdata.split_k1_into_other(idx_split, other_label)

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
def test_KData_split_k2_into_other(consistently_shaped_kdata, monkeypatch, nother_split, other_label):
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
    kdata_split = consistently_shaped_kdata.split_k2_into_other(idx_split, other_label)

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
def test_KData_select_other_subset(consistently_shaped_kdata, monkeypatch, subset_label, subset_idx):
    """Test selection of a subset from other dimension."""
    # Create KData
    nother, ncoil, nk2, nk1, nk0 = consistently_shaped_kdata.data.shape

    # Set required parameters used in sel_kdata_subset.
    _, iother, _ = torch.meshgrid(torch.arange(nk2), torch.arange(nother), torch.arange(nk1), indexing='xy')
    monkeypatch.setattr(consistently_shaped_kdata.header.acq_info.idx, subset_label, iother)

    # Select subset of data
    kdata_subset = consistently_shaped_kdata.select_other_subset(subset_idx, subset_label)

    # Verify shape of data
    assert kdata_subset.data.shape == (subset_idx.shape[0], ncoil, nk2, nk1, nk0)
    # Verify other labe describes subset data
    assert all(torch.unique(getattr(kdata_subset.header.acq_info.idx, subset_label)) == torch.unique(subset_idx))
