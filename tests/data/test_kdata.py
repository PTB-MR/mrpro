"""Tests for the KData class."""

import pytest
import torch
from einops import repeat
from mrpro.data import KData, KTrajectory, SpatialDimension
from mrpro.data.acq_filters import has_n_coils, is_coil_calibration_acquisition, is_image_acquisition
from mrpro.data.AcqInfo import rearrange_acq_info_fields
from mrpro.data.traj_calculators.KTrajectoryCalculator import DummyTrajectory
from mrpro.operators import FastFourierOp
from mrpro.utils import split_idx

from tests import relative_image_difference
from tests.conftest import RandomGenerator, generate_random_data
from tests.data import IsmrmrdRawTestData
from tests.phantoms import EllipsePhantomTestData


@pytest.fixture(scope='session')
def ismrmrd_cart_bodycoil_and_surface_coil(ellipse_phantom, tmp_path_factory):
    """Fully sampled cartesian data set with bodycoil and surface coil data."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        phantom=ellipse_phantom.phantom,
        add_bodycoil_acquisitions=True,
    )
    return ismrmrd_kdata


@pytest.fixture(scope='session')
def ismrmrd_cart_with_calibration_lines(ellipse_phantom, tmp_path_factory):
    """Undersampled Cartesian data set with calibration lines."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=1,
        acceleration=2,
        phantom=ellipse_phantom.phantom,
        n_separate_calibration_lines=16,
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

    kheader.acq_info.apply_(
        lambda field: rearrange_acq_info_fields(
            field, '(other k2 k1) ... -> other k2 k1 ...', other=n_other, k2=n_k2, k1=n_k1
        )
    )

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


def test_KData_raise_warning_for_bodycoil(ismrmrd_cart_bodycoil_and_surface_coil):
    """Mix of bodycoil and surface coil acquisitions leads to warning."""
    with pytest.raises(UserWarning, match='Acquisitions with different number'):
        _ = KData.from_file(ismrmrd_cart_bodycoil_and_surface_coil.filename, DummyTrajectory())


@pytest.mark.filterwarnings('ignore:Acquisitions with different number:UserWarning')
def test_KData_select_bodycoil_via_filter(ismrmrd_cart_bodycoil_and_surface_coil):
    """Bodycoil can be selected via a custom acquisition filter."""
    # This is the recommended way of selecting the body coil (i.e. 2 receiver elements)
    kdata = KData.from_file(
        ismrmrd_cart_bodycoil_and_surface_coil.filename,
        DummyTrajectory(),
        acquisition_filter_criterion=lambda acq: has_n_coils(2, acq) and is_image_acquisition(acq),
    )
    assert kdata.data.shape[-4] == 2


def test_KData_raise_wrong_coil_number(ismrmrd_cart):
    """Wrong number of coils leads to empty acquisitions."""
    with pytest.raises(ValueError, match='No acquisitions meeting the given filter criteria were found'):
        _ = KData.from_file(
            ismrmrd_cart.filename,
            DummyTrajectory(),
            acquisition_filter_criterion=lambda acq: has_n_coils(2, acq) and is_image_acquisition(acq),
        )


def test_KData_from_file_diff_nky_for_rep(ismrmrd_cart_invalid_reps):
    """Multiple repetitions with different number of phase encoding lines."""
    with pytest.warns(UserWarning, match=r'different number'):
        kdata = KData.from_file(ismrmrd_cart_invalid_reps.filename, DummyTrajectory())
    assert kdata.data.shape[-2] == 1, 'k1 should be 1'
    assert kdata.data.shape[-3] == 1, 'k2 should be 1'


def test_KData_calibration_lines(ismrmrd_cart_with_calibration_lines):
    """Correct handling of calibration lines."""
    # Exclude calibration lines
    kdata = KData.from_file(ismrmrd_cart_with_calibration_lines.filename, DummyTrajectory())
    assert (
        kdata.data.shape[-2]
        == ismrmrd_cart_with_calibration_lines.matrix_size // ismrmrd_cart_with_calibration_lines.acceleration
    )

    # Get only calibration lines
    kdata = KData.from_file(
        ismrmrd_cart_with_calibration_lines.filename,
        DummyTrajectory(),
        acquisition_filter_criterion=is_coil_calibration_acquisition,
    )
    assert kdata.data.shape[-2] == ismrmrd_cart_with_calibration_lines.n_separate_calibration_lines


def test_KData_kspace(ismrmrd_cart):
    """Read in data and verify k-space by comparing reconstructed image."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    ff_op = FastFourierOp(dim=(-1, -2))
    (reconstructed_img,) = ff_op.adjoint(kdata.data)

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img[0, 0, 0, ...], ismrmrd_cart.img_ref) <= 0.05


@pytest.mark.parametrize(('field', 'value'), [('lamor_frequency_proton', 42.88 * 1e6), ('tr', torch.tensor([24.3]))])
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


@pytest.mark.cuda
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


@pytest.mark.cuda
def test_KData_to_cuda(ismrmrd_cart):
    """Test KData.to to move data to CUDA memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    cuda_device = f'cuda:{torch.cuda.current_device()}'
    kdata_cuda = kdata.to(device=cuda_device)
    assert kdata_cuda.data.is_cuda
    assert kdata_cuda.traj.kz.is_cuda
    assert kdata_cuda.traj.ky.is_cuda
    assert kdata_cuda.traj.kx.is_cuda


@pytest.mark.cuda
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


@pytest.mark.cuda
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


@pytest.mark.cuda
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
    idx_k1 = repeat(torch.arange(n_k1, dtype=torch.int32), 'k1 -> other k2 k1', other=1, k2=1)

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


def test_modify_acq_info(random_kheader_shape):
    """Test the modification of the acquisition info."""
    # Create random header where AcqInfo fields are of shape [n_k1*n_k2] and reshape to [n_other, n_k2, n_k1]
    kheader, n_other, _, n_k2, n_k1, _ = random_kheader_shape

    kheader.acq_info.apply_(
        lambda field: rearrange_acq_info_fields(
            field, '(other k2 k1) ... -> other k2 k1 ...', other=n_other, k2=n_k2, k1=n_k1
        )
    )

    # Verify shape
    assert kheader.acq_info.center_sample.shape == (n_other, n_k2, n_k1, 1)
    assert kheader.acq_info.idx.k1.shape == (n_other, n_k2, n_k1)
    assert kheader.acq_info.orientation.shape == (n_other, n_k2, n_k1, 1)
    assert kheader.acq_info.position.z.shape == (n_other, n_k2, n_k1, 1)


def test_KData_compress_coils(ismrmrd_cart):
    """Test coil combination does not alter image content (much)."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata = kdata.compress_coils(n_compressed_coils=4)
    ff_op = FastFourierOp(dim=(-1, -2))
    (reconstructed_img,) = ff_op.adjoint(kdata.data)

    # Image content of each coil is the same. Therefore we only compare one coil image but we need to normalize.
    reconstructed_img = reconstructed_img[0, 0, 0, ...].abs()
    reconstructed_img /= reconstructed_img.max()
    ref_img = ismrmrd_cart.img_ref[0, 0, 0, ...].abs()
    ref_img /= ref_img.max()

    assert relative_image_difference(reconstructed_img, ref_img) <= 0.1


@pytest.mark.parametrize(
    ('batch_dims', 'joint_dims'),
    [
        (None, ...),
        ((0,), ...),
        ((-2, -1), ...),
        (None, (-1, -2, -3)),
        (None, (0, -1, -2, -3)),
    ],
    ids=[
        'single_compression',
        'batching_along_dim0',
        'batching_along_dim-2_and_dim-1',
        'single_compression_for_last_3_dims',
        'single_compression_for_last_3_and_first_dims',
    ],
)
def test_KData_compress_coils_diff_batch_joint_dims(consistently_shaped_kdata, batch_dims, joint_dims):
    """Test that all of these options work and yield the same shape."""
    n_compressed_coils = 4
    orig_kdata_shape = consistently_shaped_kdata.data.shape
    kdata = consistently_shaped_kdata.compress_coils(n_compressed_coils, batch_dims, joint_dims)
    assert kdata.data.shape == (*orig_kdata_shape[:-4], n_compressed_coils, *orig_kdata_shape[-3:])


def test_KData_compress_coils_error_both_batch_and_joint(consistently_shaped_kdata):
    """Test if error is raised if both batch_dims and joint_dims is defined."""
    with pytest.raises(ValueError, match='Either batch_dims or joint_dims'):
        consistently_shaped_kdata.compress_coils(n_compressed_coils=3, batch_dims=(0,), joint_dims=(0,))


def test_KData_compress_coils_error_coil_dim(consistently_shaped_kdata):
    """Test if error is raised if coil_dim is in batch_dims or joint_dims."""
    with pytest.raises(ValueError, match='Coil dimension must not'):
        consistently_shaped_kdata.compress_coils(n_compressed_coils=3, batch_dims=(-4,))

    with pytest.raises(ValueError, match='Coil dimension must not'):
        consistently_shaped_kdata.compress_coils(n_compressed_coils=3, joint_dims=(-4,))
