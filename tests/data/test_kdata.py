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
from mrpro.data import SpatialDimension
from mrpro.data.traj_calculators._KTrajectoryCalculator import DummyTrajectory
from mrpro.operators import FastFourierOp
from mrpro.utils import modify_acq_info
from mrpro.utils import split_idx

from tests.conftest import RandomGenerator
from tests.conftest import generate_random_data
from tests.data import IsmrmrdRawTestData
from tests.helper import relative_image_difference
from tests.phantoms._EllipsePhantomTestData import EllipsePhantomTestData


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


def test_KData_to_float64tensor(ismrmrd_cart):
    """Change KData dtype to double using other-tensor overload."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_float64 = kdata.to(torch.ones(1, dtype=torch.float64))
    assert kdata is not kdata_float64
    assert kdata_float64.data.dtype == torch.complex128
    torch.testing.assert_close(kdata_float64.data.to(dtype=torch.complex64), kdata.data)


@pytest.mark.cuda()
def test_KData_to_cudatensor(ismrmrd_cart):
    """Move KData to cuda  using other-tensor overload."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cuda = kdata.to(torch.ones(1, device=torch.device('cuda')))
    assert kdata is not kdata_cuda
    assert kdata_cuda.data.dtype == torch.complex64
    assert kdata_cuda.data.is_cuda


def test_Kdata_to_same_copy(ismrmrd_cart):
    """Call .to with no change in dtype or device."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata2 = kdata.to(copy=True)
    assert kdata is not kdata2
    assert torch.equal(kdata.data, kdata2.data)
    assert kdata2.data.dtype == kdata.data.dtype
    assert kdata2.data.device == kdata.data.device


def test_Kdata_to_same_nocopy(ismrmrd_cart):
    """Call .to with no change in dtype or device."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata2 = kdata.to(copy=False)
    assert kdata is not kdata2
    assert kdata.data is kdata2.data


def test_KData_to_complex128_data(ismrmrd_cart):
    """Change KData dtype complex128: test data."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata is not kdata_complex128
    assert kdata_complex128.data.dtype == torch.complex128
    torch.testing.assert_close(kdata_complex128.data.to(dtype=torch.complex64), kdata.data)


def test_KData_to_complex128_traj(ismrmrd_cart):
    """Change KData dtype complex128: test trajectory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata_complex128.traj.kx.dtype == torch.float64
    assert kdata_complex128.traj.ky.dtype == torch.float64
    assert kdata_complex128.traj.kz.dtype == torch.float64


def test_KData_to_complex128_header(ismrmrd_cart):
    """Change KData dtype complex128: test header"""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata_complex128.header.acq_info.user_float.dtype == torch.float64
    assert kdata_complex128.header.acq_info.user_int.dtype == torch.int32


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
    assert kdata_cuda.header.acq_info.user_int.is_cuda
    assert kdata_cuda.device == torch.device(torch.cuda.current_device())
    assert kdata_cuda.header.acq_info.device == torch.device(torch.cuda.current_device())
    assert kdata_cuda.is_cuda
    assert not kdata_cuda.is_cpu


@pytest.mark.cuda()
def test_KData_cpu(ismrmrd_cart):
    """Move KData object to CUDA memory and back to CPU memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cpu = kdata.cuda().cpu()
    assert kdata_cpu.data.is_cpu
    assert kdata_cpu.traj.kz.is_cpu
    assert kdata_cpu.traj.ky.is_cpu
    assert kdata_cpu.traj.kx.is_cpu
    assert kdata_cpu.header.acq_info.user_int.is_cpu
    assert kdata_cpu.device == torch.device('cpu')
    assert kdata_cpu.header.acq_info.device == torch.device('cpu')


def test_Kdata_device_cpu(ismrmrd_cart):
    """Default device is CPU."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert kdata.device == torch.device('cpu')
    assert not kdata.is_cuda
    assert kdata.is_cpu


@pytest.mark.cuda()
def test_KData_inconsistentdevice(ismrmrd_cart):
    """Inconsistent device raises exception."""
    kdata_cpu = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cuda = kdata_cpu.to(device='cuda')
    kdata_mix = KData(data=kdata_cuda.data, header=kdata_cpu.header, traj=kdata_cpu.traj)
    assert not kdata_mix.is_cuda
    assert not kdata_mix.is_cpu
    with pytest.raises(ValueError):
        _ = kdata_mix.device


def test_KData_clone(ismrmrd_cart):
    """Test .clone method."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata2 = kdata.clone()
    assert kdata is not kdata2
    assert kdata.data is not kdata2.data
    assert torch.equal(kdata.data, kdata2.data)
    assert kdata.traj.kx is not kdata2.traj.kx
    assert torch.equal(kdata.traj.kx, kdata2.traj.kx)


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


def test_KData_remove_readout_os(monkeypatch, random_kheader):
    # Dimensions
    n_coils = 4
    n_k0 = 240
    n_k1 = 240
    n_k0_oversampled = 320
    discard_pre = 10
    discard_post = 20

    random_generator = RandomGenerator(seed=0)

    # List of k1 indices in the shape
    idx_k1 = torch.arange(n_k1, dtype=torch.int32)[None, None, ...]

    # Set parameters need in remove_os
    monkeypatch.setattr(random_kheader.encoding_matrix, 'x', n_k0_oversampled)
    monkeypatch.setattr(random_kheader.recon_matrix, 'x', n_k0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1) + n_k0_oversampled // 2)
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1) + n_k0_oversampled)
    monkeypatch.setattr(random_kheader.acq_info, 'discard_pre', torch.tensor(discard_pre, dtype=torch.int32))
    monkeypatch.setattr(random_kheader.acq_info, 'discard_post', torch.tensor(discard_post, dtype=torch.int32))

    # Create kspace and image with oversampling
    phantom_os = EllipsePhantomTestData(n_y=n_k1, n_x=n_k0_oversampled)
    kdata_os = phantom_os.phantom.kspace(phantom_os.ky, phantom_os.kx)
    img_dim = SpatialDimension(z=1, y=n_k1, x=n_k0_oversampled)
    img_tensor = phantom_os.phantom.image_space(img_dim)

    # Crop image data
    start = (n_k0_oversampled - n_k0) // 2
    img_tensor = img_tensor[..., start : start + n_k0]

    # Create k-space data with correct dimensions
    k_tensor = repeat(kdata_os, 'k1 k0 -> other coils k2 k1 k0', other=1, coils=n_coils, k2=1)

    # Create random 2D Cartesian trajectory
    kx = random_generator.float32_tensor(size=(1, 1, 1, n_k0_oversampled))
    ky = random_generator.float32_tensor(size=(1, 1, n_k1, 1))
    kz = random_generator.float32_tensor(size=(1, 1, 1, 1))
    trajectory = KTrajectory(kz, ky, kx)

    # Create KData
    kdata = KData(header=random_kheader, data=k_tensor, traj=trajectory)

    # Remove oversampling
    kdata_without_os = kdata.remove_readout_os()

    # Reconstruct image from k-space data of one coil and compare to phantom image
    fourier_op = FastFourierOp(dim=(-1, -2))
    (img_recon,) = fourier_op.adjoint(kdata_without_os.data[:, 0, ...])

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(torch.abs(img_recon), img_tensor[:, 0, ...]) <= 0.05
