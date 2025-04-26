"""Tests for the KData class."""

import re
from collections.abc import Sequence
from types import EllipsisType
from typing import Literal

import pytest
import torch
from einops import repeat
from mrpro.data import KData, KHeader, KTrajectory, SpatialDimension
from mrpro.data.acq_filters import has_n_coils, is_coil_calibration_acquisition, is_image_acquisition
from mrpro.data.traj_calculators.KTrajectoryCalculator import DummyTrajectory
from mrpro.operators import FastFourierOp
from mrpro.utils import RandomGenerator, split_idx

from tests import relative_image_difference
from tests.phantoms import EllipsePhantomTestData


def test_KData_from_file(ismrmrd_cart) -> None:
    """Read in data from file."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert kdata is not None


def test_KData_random_cart_undersampling(ismrmrd_cart_random_us) -> None:
    """Read data with different random Cartesian undersampling in multiple
    repetitions."""
    kdata = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    assert kdata is not None


def test_KData_random_cart_undersampling_shape(ismrmrd_cart_random_us) -> None:
    """Check shape of KData with random Cartesian undersampling."""
    kdata = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    # check if the number of repetitions is correct
    assert kdata.data.shape[-5] == ismrmrd_cart_random_us.repetitions
    # check if the number of phase encoding lines per repetition is correct
    assert kdata.data.shape[-2] == ismrmrd_cart_random_us.matrix_size // ismrmrd_cart_random_us.acceleration


def test_KData_raise_wrong_trajectory_shape(ismrmrd_cart) -> None:
    """Wrong KTrajectory shape raises exception."""
    rng = RandomGenerator(seed=0)
    trajectory = KTrajectory(*rng.float32_tensor((3, 5, 1, 2, 3, 4)))
    with pytest.raises(ValueError):
        _ = KData.from_file(ismrmrd_cart.filename, trajectory)


def test_KData_raise_warning_for_bodycoil(ismrmrd_cart_bodycoil_and_surface_coil) -> None:
    """Mix of bodycoil and surface coil acquisitions leads to warning."""
    with pytest.raises(UserWarning, match='Acquisitions with different number'):
        _ = KData.from_file(ismrmrd_cart_bodycoil_and_surface_coil.filename, DummyTrajectory())


@pytest.mark.filterwarnings('ignore:Acquisitions with different number:UserWarning')
def test_KData_select_bodycoil_via_filter(ismrmrd_cart_bodycoil_and_surface_coil) -> None:
    """Bodycoil can be selected via a custom acquisition filter."""
    # This is the recommended way of selecting the body coil (i.e. 2 receiver elements)
    kdata = KData.from_file(
        ismrmrd_cart_bodycoil_and_surface_coil.filename,
        DummyTrajectory(),
        acquisition_filter_criterion=lambda acq: has_n_coils(2, acq) and is_image_acquisition(acq),
    )
    assert kdata.data.shape[-4] == 2


def test_KData_raise_wrong_coil_number(ismrmrd_cart) -> None:
    """Wrong number of coils leads to empty acquisitions."""
    with pytest.raises(ValueError, match='No acquisitions meeting the given filter criteria were found'):
        _ = KData.from_file(
            ismrmrd_cart.filename,
            DummyTrajectory(),
            acquisition_filter_criterion=lambda acq: has_n_coils(2, acq) and is_image_acquisition(acq),
        )


def test_KData_from_file_diff_nky_for_rep(ismrmrd_cart_invalid_reps) -> None:
    """Multiple repetitions with different number of phase encoding lines."""
    with pytest.warns(UserWarning, match=r'different number'):
        kdata = KData.from_file(ismrmrd_cart_invalid_reps.filename, DummyTrajectory())
    assert kdata.data.shape[-2] == 1, 'k1 should be 1'
    assert kdata.data.shape[-3] == 1, 'k2 should be 1'


def test_KData_calibration_lines(ismrmrd_cart_with_calibration_lines) -> None:
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


def test_KData_kspace(ismrmrd_cart_high_res) -> None:
    """Read in data and verify k-space by comparing reconstructed image."""
    kdata = KData.from_file(ismrmrd_cart_high_res.filename, DummyTrajectory())
    ff_op = FastFourierOp(
        dim=(-1, -2),
        recon_matrix=[kdata.header.recon_matrix.x, kdata.header.recon_matrix.y],
        encoding_matrix=[kdata.header.encoding_matrix.x, kdata.header.encoding_matrix.y],
    )
    (reconstructed_img,) = ff_op.adjoint(kdata.data)

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img[0, 0, 0, ...], ismrmrd_cart_high_res.img_ref) <= 0.05


@pytest.mark.parametrize(('field', 'value'), [('lamor_frequency_proton', 42.88 * 1e6), ('tr', [24.3])])
def test_KData_overwrite_header(ismrmrd_cart, field: str, value: float) -> None:
    """Overwrite some parameters in the header."""
    parameter_dict = {field: value}
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory(), header_overwrites=parameter_dict)
    assert getattr(kdata.header, field) == value


def test_KData_to_float64tensor(ismrmrd_cart) -> None:
    """Change KData dtype to double using other-tensor overload."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_float64 = kdata.to(torch.ones(1, dtype=torch.float64))
    assert kdata is not kdata_float64
    assert kdata_float64.data.dtype == torch.complex128
    torch.testing.assert_close(kdata_float64.data.to(dtype=torch.complex64), kdata.data)


@pytest.mark.cuda
def test_KData_to_cudatensor(ismrmrd_cart) -> None:
    """Move KData to cuda  using other-tensor overload."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cuda = kdata.to(torch.ones(1, device=torch.device('cuda')))
    assert kdata is not kdata_cuda
    assert kdata_cuda.data.dtype == torch.complex64
    assert kdata_cuda.data.is_cuda


def test_Kdata_to_same_copy(ismrmrd_cart) -> None:
    """Call .to with no change in dtype or device."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata2 = kdata.to(copy=True)
    assert kdata is not kdata2
    assert torch.equal(kdata.data, kdata2.data)
    assert kdata2.data.dtype == kdata.data.dtype
    assert kdata2.data.device == kdata.data.device


def test_Kdata_to_same_nocopy(ismrmrd_cart) -> None:
    """Call .to with no change in dtype or device."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata2 = kdata.to(copy=False)
    assert kdata is not kdata2
    assert kdata.data is kdata2.data


def test_KData_to_complex128_data(ismrmrd_cart) -> None:
    """Change KData dtype complex128: test data."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata is not kdata_complex128
    assert kdata_complex128.data.dtype == torch.complex128
    torch.testing.assert_close(kdata_complex128.data.to(dtype=torch.complex64), kdata.data)


def test_KData_to_complex128_traj(ismrmrd_cart) -> None:
    """Change KData dtype complex128: test trajectory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata_complex128.traj.kx.dtype == torch.float64
    assert kdata_complex128.traj.ky.dtype == torch.float64
    assert kdata_complex128.traj.kz.dtype == torch.float64


def test_KData_to_complex128_header(ismrmrd_cart) -> None:
    """Change KData dtype complex128: test header"""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata_complex128.header.acq_info.user.float1.dtype == torch.float64
    assert kdata_complex128.header.acq_info.user.int1.dtype == torch.int32


@pytest.mark.cuda
def test_KData_to_cuda(ismrmrd_cart) -> None:
    """Test KData.to to move data to CUDA memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    cuda_device = f'cuda:{torch.cuda.current_device()}'
    kdata_cuda = kdata.to(device=cuda_device)
    assert kdata_cuda.data.is_cuda
    assert kdata_cuda.traj.kz.is_cuda
    assert kdata_cuda.traj.ky.is_cuda
    assert kdata_cuda.traj.kx.is_cuda


@pytest.mark.cuda
def test_KData_cuda(ismrmrd_cart) -> None:
    """Move KData object to CUDA memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cuda = kdata.cuda()
    assert kdata_cuda.data.is_cuda
    assert kdata_cuda.traj.kz.is_cuda
    assert kdata_cuda.traj.ky.is_cuda
    assert kdata_cuda.traj.kx.is_cuda
    assert kdata_cuda.header.acq_info.user.int1.is_cuda
    assert kdata_cuda.device == torch.device(torch.cuda.current_device())
    assert kdata_cuda.header.acq_info.device == torch.device(torch.cuda.current_device())
    assert kdata_cuda.is_cuda
    assert not kdata_cuda.is_cpu


@pytest.mark.cuda
def test_KData_cpu(ismrmrd_cart) -> None:
    """Move KData object to CUDA memory and back to CPU memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cpu = kdata.cuda().cpu()
    assert kdata_cpu.data.is_cpu
    assert kdata_cpu.traj.kz.is_cpu
    assert kdata_cpu.traj.ky.is_cpu
    assert kdata_cpu.traj.kx.is_cpu
    assert kdata_cpu.header.acq_info.user.int1.is_cpu
    assert kdata_cpu.device == torch.device('cpu')
    assert kdata_cpu.header.acq_info.device == torch.device('cpu')


def test_Kdata_device_cpu(ismrmrd_cart) -> None:
    """Default device is CPU."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert kdata.device == torch.device('cpu')
    assert not kdata.is_cuda
    assert kdata.is_cpu


@pytest.mark.cuda
def test_KData_inconsistentdevice(ismrmrd_cart) -> None:
    """Inconsistent device raises exception."""
    kdata_cpu = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cuda = kdata_cpu.to(device='cuda')
    kdata_mix = KData(data=kdata_cuda.data, header=kdata_cpu.header, traj=kdata_cpu.traj)
    assert not kdata_mix.is_cuda
    assert not kdata_mix.is_cpu
    with pytest.raises(ValueError):
        _ = kdata_mix.device


def test_KData_clone(ismrmrd_cart) -> None:
    """Test .clone method."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata2 = kdata.clone()
    assert kdata is not kdata2
    assert kdata.data is not kdata2.data
    assert torch.equal(kdata.data, kdata2.data)
    assert kdata.traj.kx is not kdata2.traj.kx
    assert torch.equal(kdata.traj.kx, kdata2.traj.kx)


@pytest.mark.parametrize(
    ('n_other_split', 'other_label'),
    [
        (10, 'average'),
        (5, 'repetition'),
        (7, 'contrast'),
    ],
)
def test_KData_split_k1_into_other(
    consistently_shaped_kdata: KData, n_other_split: int, other_label: Literal['average', 'repetition', 'contrast']
) -> None:
    """Test splitting of the k1 dimension into other."""
    *n_others, n_coils, n_k2, n_k1, n_k0 = consistently_shaped_kdata.data.shape
    n_other = torch.tensor(n_others).prod().item()

    # Split index
    k1_per_block = n_k1 // n_other_split
    idx_k1 = torch.linspace(0, n_k1 - 1, n_k1, dtype=torch.int32)
    idx_split = split_idx(idx_k1, k1_per_block)

    # Split data
    kdata_split = consistently_shaped_kdata.split_k1_into_other(idx_split, other_label)

    assert kdata_split.data.shape == (idx_split.shape[0] * n_other, n_coils, n_k2, k1_per_block, n_k0)
    assert kdata_split.traj.shape == (idx_split.shape[0] * n_other, 1, n_k2, k1_per_block, n_k0)
    new_idx = getattr(kdata_split.header.acq_info.idx, other_label)
    assert new_idx.shape == (idx_split.shape[0] * n_other, 1, 1, 1, 1)


@pytest.mark.parametrize(
    ('subset_label', 'subset_idx'),
    [
        ('repetition', torch.tensor([1], dtype=torch.int32)),
        ('average', torch.tensor([0, 1], dtype=torch.int32)),
        ('phase', torch.tensor([1, 0, 0], dtype=torch.int32)),
    ],
)
def test_KData_select_other_subset(
    consistently_shaped_kdata: KData,
    monkeypatch,
    subset_label: Literal['repetition', 'average', 'phase'],
    subset_idx: torch.Tensor,
) -> None:
    """Test selection of a subset from other dimension."""
    # Create KData
    *n_other, n_coils, n_k2, n_k1, n_k0 = consistently_shaped_kdata.data.shape

    # Set required parameters used in sel_kdata_subset.
    idx = (
        torch.arange(torch.tensor(n_other).prod().item()).view(*n_other, 1, 1, 1, 1).expand(*n_other, 1, n_k2, n_k1, 1)
    )
    monkeypatch.setattr(consistently_shaped_kdata.header.acq_info.idx, subset_label, idx)

    # Select subset of data
    kdata_subset = consistently_shaped_kdata.select_other_subset(subset_idx, subset_label)

    # Verify shape of data
    assert kdata_subset.data.shape == (subset_idx.shape[0], n_coils, n_k2, n_k1, n_k0)
    # Verify other label describes subset data
    assert all(torch.unique(getattr(kdata_subset.header.acq_info.idx, subset_label)) == torch.unique(subset_idx))


def test_KData_remove_readout_os(monkeypatch, random_kheader: KHeader) -> None:
    # Dimensions
    n_coils = 4
    n_k0 = 240
    n_k1 = 240
    n_k0_oversampled = 320

    rng = RandomGenerator(seed=0)

    # Set parameters need in remove_os
    monkeypatch.setattr(random_kheader.encoding_matrix, 'x', n_k0_oversampled)
    monkeypatch.setattr(random_kheader.recon_matrix, 'x', n_k0)

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
    kx = rng.float32_tensor(size=(1, 1, 1, 1, n_k0_oversampled))
    ky = rng.float32_tensor(size=(1, 1, 1, n_k1, 1))
    kz = rng.float32_tensor(size=(1, 1, 1, 1, 1))
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


def test_KData_compress_coils(ismrmrd_cart_high_res) -> None:
    """Test coil combination does not alter image content (much)."""
    kdata = KData.from_file(ismrmrd_cart_high_res.filename, DummyTrajectory())
    kdata = kdata.compress_coils(n_compressed_coils=4)
    ff_op = FastFourierOp(
        dim=(-1, -2),
        recon_matrix=[kdata.header.recon_matrix.x, kdata.header.recon_matrix.y],
        encoding_matrix=[kdata.header.encoding_matrix.x, kdata.header.encoding_matrix.y],
    )
    (reconstructed_img,) = ff_op.adjoint(kdata.data)

    # Image content of each coil is the same. Therefore we only compare one coil image but we need to normalize.
    reconstructed_img = reconstructed_img[0, 0, 0, ...].abs()
    reconstructed_img /= reconstructed_img.max()
    ref_img = ismrmrd_cart_high_res.img_ref[0, 0, 0, ...].abs()
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
def test_KData_compress_coils_diff_batch_joint_dims(
    consistently_shaped_kdata: KData, batch_dims: Sequence[int], joint_dims: Sequence[int] | EllipsisType
) -> None:
    """Test that all of these options work and yield the same shape."""
    n_compressed_coils = 2
    orig_kdata_shape = consistently_shaped_kdata.data.shape
    kdata = consistently_shaped_kdata.compress_coils(n_compressed_coils, batch_dims, joint_dims)
    assert kdata.data.shape == (*orig_kdata_shape[:-4], n_compressed_coils, *orig_kdata_shape[-3:])


def test_KData_compress_coils_error_both_batch_and_joint(consistently_shaped_kdata: KData) -> None:
    """Test if error is raised if both batch_dims and joint_dims is defined."""
    with pytest.raises(ValueError, match='Either batch_dims or joint_dims'):
        consistently_shaped_kdata.compress_coils(n_compressed_coils=3, batch_dims=(0,), joint_dims=(0,))


def test_KData_compress_coils_error_coil_dim(consistently_shaped_kdata: KData) -> None:
    """Test if error is raised if coil_dim is in batch_dims or joint_dims."""
    with pytest.raises(ValueError, match='Coil dimension must not'):
        consistently_shaped_kdata.compress_coils(n_compressed_coils=3, batch_dims=(-4,))

    with pytest.raises(ValueError, match='Coil dimension must not'):
        consistently_shaped_kdata.compress_coils(n_compressed_coils=3, joint_dims=(-4,))


def test_KData_compress_coils_error_n_coils(consistently_shaped_kdata: KData) -> None:
    """Test if error is raised if new coils would be larger than existing coils"""
    existing_coils = consistently_shaped_kdata.data.shape[-4]
    with pytest.raises(ValueError, match='greater'):
        consistently_shaped_kdata.compress_coils(existing_coils + 1)


def test_KData_repr(consistently_shaped_kdata: KData) -> None:
    actual_repr = repr(consistently_shaped_kdata)
    actual_str = str(consistently_shaped_kdata)
    assert actual_str == actual_repr
    assert re.match(
        r"""KData on device "cpu" with \(broadcasted\) shape \[2, 3, 3, 13, 11, 10\]\.
\s{2}data: Tensor<2, 3, 3, 13, 11, 10>, \|x\| ∈ \[(.*?),\s*(.*?)\]\s*,\s*μ=(.*?),\s*\[(.*?),\s*\.\.\.,\s*(.*?)\]
\s{2}traj: KTrajectory on device "cpu" with \(broadcasted\) shape \[2, 1, 1, 13, 11, 10\]\.
\s{5}kz: Tensor<2, 1, 1, 13, 1, 1>, x ∈ \[(.*?),\s*(.*?)\]\s*,\s*μ=(.*?),\s*\[(.*?),\s*\.\.\.,\s*(.*?)\]
\s{5}ky: Tensor<2, 1, 1, 1, 11, 1>, x ∈ \[(.*?),\s*(.*?)\]\s*,\s*μ=(.*?),\s*\[\s*(.*?),\s*\.\.\.,\s*(.*?)\s*\]
\s{5}kx: Tensor<2, 1, 1, 1, 1, 10>, x ∈ \[(.*?),\s*(.*?)\]\s*,\s*μ=(.*?),\s*\[(.*?),\s*\.\.\.,\s*(.*?)\]
\s{5}grid_detection_tolerance:\s*(.*?)
\s{2}header:\s{2}KHeader on device "cpu" with \(broadcasted\) shape \[2, 1, 1, 13, 11, 1\]\.
\s{5}recon_matrix: z=17, y=83, x=47
\s{5}encoding_matrix: z=73, y=81, x=66
\s{5}recon_fov: z=(.*?),\s*y=(.*?),\s*x=(.*?)
\s{5}encoding_fov: z=(.*?),\s*y=(.*?),\s*x=(.*?)
\s{5}acq_info: AcqInfo<2, 1, 1, 13, 11, 1>
\s{5}te: list
\s{5}ti: list
\s{5}fa: list
\s{5}tr: list
\s{5}echo_spacing: list
\s{5}echo_train_length: 1
\s{5}sequence_type: unknown
\s{5}model: unknown
\s{5}vendor: unknown
\s{5}protocol_name: unknown
\s{5}trajectory_type: TrajectoryType\.OTHER
\s{5}measurement_id: unknown
\s{5}patient_name: unknown""",
        actual_repr,
    )
