"""Tests for Non-Uniform Fast Fourier operator."""

import einops
import pytest
import torch
from mrpro.data import DcfData, KData, KTrajectory, SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import DensityCompensationOp, FastFourierOp, NonUniformFastFourierOp, PCACompressionOp
from mrpro.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp
from mrpro.utils import RandomGenerator
from torch.autograd.gradcheck import gradcheck

from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.helper import dotproduct_adjointness_test, relative_image_difference


def create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz) -> tuple[torch.Tensor, KTrajectory]:
    """Create k-space trajectory and random image."""
    rng = RandomGenerator(seed=0)
    img = rng.complex64_tensor(size=img_shape)
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


def create_time_varying_2d_nufft_op(
    n_timepoints: int = 4,
    image_shape: tuple[int, int] = (12, 10),
) -> NonUniformFastFourierOp:
    """Create a small time-varying 2D NUFFT over y/x."""
    nk = (n_timepoints, 1, 1, *image_shape)
    trajectory = create_traj(
        nkx=nk,
        nky=nk,
        nkz=(n_timepoints, 1, 1, 1, 1),
        type_kx='non-uniform',
        type_ky='non-uniform',
        type_kz='zero',
    )
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    return NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=SpatialDimension(z=1, y=image_shape[0], x=image_shape[1]),
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )


@COMMON_MR_TRAJECTORIES
def test_non_uniform_fast_fourier_op_fwd_adj_property(
    img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test adjoint property of non-uniform Fast Fourier operator."""

    # generate random images and k-space trajectories
    _, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=img_shape)
    v = rng.complex64_tensor(size=k_shape)
    dotproduct_adjointness_test(nufft_op, u, v)


@COMMON_MR_TRAJECTORIES
def test_non_uniform_fast_fourier_op_gram(
    img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test gram of of non-uniform Fast Fourier operator."""
    img, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )

    (expected,) = (nufft_op.H @ nufft_op)(img)
    (actual,) = nufft_op.gram(img)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_non_uniform_fast_fourier_op_equal_to_fft(ismrmrd_cart_high_res) -> None:
    """Compare nufft result to fft result for Cartesian data."""
    kdata = KData.from_file(ismrmrd_cart_high_res.filename, KTrajectoryIsmrmrd())
    nufft_op = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=[kdata.header.recon_matrix.y, kdata.header.recon_matrix.x],
        encoding_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
        traj=kdata.traj,
    )
    (nufft_img,) = nufft_op.adjoint(kdata.data)
    fft_op = FastFourierOp(
        dim=(-2, -1),
        recon_matrix=[kdata.header.recon_matrix.y, kdata.header.recon_matrix.x],
        encoding_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
    )
    (fft_img,) = fft_op.adjoint(kdata.data)
    torch.testing.assert_close(nufft_img, fft_img, rtol=1e-3, atol=1e-3)


def test_non_uniform_fast_fourier_cartesian_result(ismrmrd_cart_high_res) -> None:
    """Eval result of non-uniform Fast Fourier transform for Cartesian data."""
    kdata = KData.from_file(ismrmrd_cart_high_res.filename, KTrajectoryIsmrmrd())
    nufft_op = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=[kdata.header.recon_matrix.y, kdata.header.recon_matrix.x],
        encoding_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
        traj=kdata.traj,
    )
    (reconstructed_img,) = nufft_op.adjoint(kdata.data)
    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img[0, 0, 0, ...], ismrmrd_cart_high_res.img_ref) <= 0.05


def test_non_uniform_fast_fourier_op_empty_dims() -> None:
    """Empty dims do not change the input."""
    nk = (1, 1, 1, 1, 1)
    traj = create_traj(nkx=nk, nky=nk, nkz=nk, type_kx='non-uniform', type_ky='non-uniform', type_kz='non-uniform')
    nufft_op = NonUniformFastFourierOp(direction=(), recon_matrix=(), encoding_matrix=(), traj=traj)

    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=[2, 3, 4, 5])
    torch.testing.assert_close(u, nufft_op(u)[0])


def test_non_uniform_fast_fourier_op_directions() -> None:
    """Test different direction specifiers of non-uniform fast Fourier operator."""

    kdata_shape = (1, 3, 1, 30, 40)
    img_shape = (2, 3, 1, 20, 30)

    # generate random traj and image
    nk = [2, 1, 1, 30, 40]
    img, traj = create_data(
        img_shape,
        nkx=nk,
        nky=nk,
        nkz=nk,
        type_kx='non-uniform',
        type_ky='non-uniform',
        type_kz='non-uniform',
    )

    # create operator
    nufft_op_12 = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=SpatialDimension[int](z=img_shape[-3], y=img_shape[-2], x=img_shape[-1]),
        encoding_matrix=SpatialDimension[int](z=kdata_shape[-3], y=kdata_shape[-2], x=kdata_shape[-1]),
        traj=traj,
    )

    nufft_op_yx = NonUniformFastFourierOp(
        direction=('y', 'x'),
        recon_matrix=SpatialDimension[int](z=img_shape[-3], y=img_shape[-2], x=img_shape[-1]),
        encoding_matrix=SpatialDimension[int](z=kdata_shape[-3], y=kdata_shape[-2], x=kdata_shape[-1]),
        traj=traj,
    )

    nufft_op_xy = NonUniformFastFourierOp(
        direction=('x', 'y'),
        recon_matrix=SpatialDimension[int](z=img_shape[-3], y=img_shape[-2], x=img_shape[-1]),
        encoding_matrix=SpatialDimension[int](z=kdata_shape[-3], y=kdata_shape[-2], x=kdata_shape[-1]),
        traj=traj,
    )

    nufft_op_xy_matrix_sequence = NonUniformFastFourierOp(
        direction=('x', 'y'),
        recon_matrix=(img_shape[-1], img_shape[-2]),
        encoding_matrix=(kdata_shape[-1], kdata_shape[-2]),
        traj=traj,
    )
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_yx(img)[0])
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_xy(img)[0])
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_xy_matrix_sequence(img)[0])


def test_non_uniform_fast_fourier_op_error_directions() -> None:
    """Test error for duplicate directions of non-uniform fast Fourier operator."""
    with pytest.raises(ValueError, match='Directions must be unique'):
        NonUniformFastFourierOp(
            direction=(-2, -1, 'x'),
            recon_matrix=[1, 1],
            encoding_matrix=[1, 1],
            traj=KTrajectory.from_tensor(torch.ones((3, 1, 1, 1, 1, 1))),
        )


def test_non_uniform_fast_fourier_op_error_matrix() -> None:
    """Test error for wrong matrix dimensions of non-uniform fast Fourier operator."""
    with pytest.raises(ValueError, match='recon_matrix should have'):
        NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=[1, 1, 1],
            encoding_matrix=[1, 1],
            traj=KTrajectory.from_tensor(torch.ones((3, 1, 1, 1, 1, 1))),
        )

    with pytest.raises(ValueError, match='encoding_matrix should have'):
        NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=[1, 1],
            encoding_matrix=[1],
            traj=KTrajectory.from_tensor(torch.ones((3, 1, 1, 1, 1, 1))),
        )


def test_non_uniform_fast_fourier_op_repr():
    """Test the __repr__ method of NonUniformFastFourierOp."""

    # Create a trajectory with non-Cartesian (off-grid) components
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 1, 60)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 20, 1, 1)
    type_kx = 'non-uniform'
    type_ky = 'non-uniform'
    type_kz = 'non-uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    repr_str = repr(nufft_op)

    # Check if __repr__ contains expected information
    assert 'NonUniformFastFourierOp' in repr_str
    assert 'Dimension(s) along which NUFFT is applied' in repr_str
    assert 'Reconstructed image size' in repr_str
    assert 'device' in repr_str


def test_subspace_non_uniform_fast_fourier_op_gram() -> None:
    """Test subspace Toeplitz Gram against explicit expand-apply-compress reference."""
    rng = RandomGenerator(seed=1)
    n_timepoints, n_coefficients = 4, 2
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    basis = rng.complex64_tensor((n_timepoints, n_coefficients))
    alpha = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape))

    subspace_gram = nufft_op.toeplitz(subspace=basis)
    assert isinstance(subspace_gram, SubspaceNonUniformFastFourierOpGramOp)

    expanded = einops.einsum(basis, alpha, 'time coeff, coeff joint coil ... -> time joint coil ...')
    (kspace,) = nufft_op(expanded)
    (backprojected,) = nufft_op.H(kspace)
    expected = einops.einsum(basis.conj(), backprojected, 'time coeff, time joint coil ... -> coeff joint coil ...')
    (actual,) = subspace_gram(alpha)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_subspace_non_uniform_fast_fourier_op_gram_non_contiguous_directions() -> None:
    """Test subspace Toeplitz Gram over non-contiguous z/x image axes."""
    rng = RandomGenerator(seed=7)
    n_coefficients = 2
    img_shape = (8, 5, 64, 1, 64)
    nkx = (8, 1, 1, 18, 128)
    nky = (8, 1, 1, 1, 1)
    nkz = (8, 1, 1, 18, 128)
    trajectory = create_traj(nkx, nky, nkz, type_kx='non-uniform', type_ky='zero', type_kz='non-uniform')

    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    basis = rng.complex64_tensor((img_shape[0], n_coefficients))
    alpha = rng.complex64_tensor((n_coefficients, *img_shape[1:]))

    subspace_gram = nufft_op.toeplitz(subspace=basis)

    expanded = einops.einsum(basis, alpha, 'time coeff, coeff coil ... -> time coil ...')
    (kspace,) = nufft_op(expanded)
    (backprojected,) = nufft_op.H(kspace)
    expected = einops.einsum(basis.conj(), backprojected, 'time coeff, time coil ... -> coeff coil ...')
    (actual,) = subspace_gram(alpha)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize('n_coefficients', [1, 2])
def test_subspace_non_uniform_fast_fourier_op_gram_single_timepoint_basis(n_coefficients: int) -> None:
    """Test single-timepoint bases keep their time and coefficient axes."""
    rng = RandomGenerator(seed=8)
    n_timepoints = 1
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    basis = rng.complex64_tensor((n_timepoints, n_coefficients))
    alpha = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape))

    expanded = einops.einsum(basis, alpha, 'time coeff, coeff joint coil ... -> time joint coil ...')
    (kspace,) = nufft_op(expanded)
    (backprojected,) = nufft_op.H(kspace)
    expected = einops.einsum(basis.conj(), backprojected, 'time coeff, time joint coil ... -> coeff joint coil ...')
    (actual,) = nufft_op.toeplitz(subspace=basis)(alpha)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_subspace_non_uniform_fast_fourier_op_gram_accepts_pca_operator() -> None:
    """Test subspace Gram accepts a PCACompressionOp as basis input."""
    rng = RandomGenerator(seed=2)
    n_timepoints, n_coefficients = 4, 2
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    training_signals = rng.complex64_tensor((1, 32, n_timepoints))
    pca_op = PCACompressionOp(training_signals, n_components=n_coefficients, centering=False)
    basis = pca_op.compression_matrix.squeeze(0).mH
    alpha = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape))

    subspace_gram_from_pca = SubspaceNonUniformFastFourierOpGramOp(nufft_op, pca_op)
    subspace_gram_from_basis = SubspaceNonUniformFastFourierOpGramOp(nufft_op, basis)

    (actual_from_pca,) = subspace_gram_from_pca(alpha)
    (actual_from_basis,) = subspace_gram_from_basis(alpha)

    torch.testing.assert_close(actual_from_pca, actual_from_basis)


def test_non_uniform_fast_fourier_op_weighted_toeplitz_matches_explicit_dcf_normal_operator() -> None:
    """Test Toeplitz(weight=dcf) matches the explicit weighted normal operator F^H DCF F."""
    rng = RandomGenerator(seed=6)
    n_timepoints = 4
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    image = rng.complex64_tensor((n_timepoints, 1, 1, *image_shape))
    dcf = DcfData(data=rng.float32_tensor((n_timepoints, 1, 1, 1, image_shape[-1])))

    explicit_operator = nufft_op.H @ DensityCompensationOp(dcf) @ nufft_op
    weighted_toeplitz = nufft_op.toeplitz(weight=dcf)

    (expected,) = explicit_operator(image)
    (actual,) = weighted_toeplitz(image)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_non_uniform_fast_fourier_op_gram_autograd() -> None:
    """Test autograd of the Toeplitz Gram operator."""
    rng = RandomGenerator(seed=3)
    n_timepoints = 2
    image_shape = (5, 4)
    nufft_op = create_time_varying_2d_nufft_op(
        n_timepoints=n_timepoints,
        image_shape=image_shape,
    )
    operator = nufft_op.toeplitz().double()
    image = rng.complex128_tensor((n_timepoints, 1, 1, *image_shape)).requires_grad_(True)

    gradcheck(operator, (image,), fast_mode=True)


def test_subspace_non_uniform_fast_fourier_op_gram_autograd() -> None:
    """Test autograd of the subspace Toeplitz Gram operator."""
    rng = RandomGenerator(seed=4)
    n_timepoints, n_coefficients = 3, 2
    image_shape = (5, 4)
    nufft_op = create_time_varying_2d_nufft_op(
        n_timepoints=n_timepoints,
        image_shape=image_shape,
    )
    basis = rng.complex128_tensor((n_timepoints, n_coefficients))
    operator = nufft_op.toeplitz(subspace=basis).double()
    coefficients = rng.complex128_tensor((n_coefficients, 1, 1, *image_shape)).requires_grad_(True)

    gradcheck(operator, (coefficients,), fast_mode=True)


@pytest.mark.cuda
def test_non_uniform_fast_fourier_op_gram_cuda() -> None:
    """Test Toeplitz Gram operators work on CUDA devices."""
    rng = RandomGenerator(seed=5)
    n_timepoints = 4
    image_shape = (12, 10)
    image = rng.complex64_tensor((n_timepoints, 1, 1, *image_shape)).cuda()

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape).cuda()
    gram = nufft_op.gram.cuda()
    (result,) = gram(image)
    assert result.is_cuda

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    gram = nufft_op.gram
    (result,) = gram(image)
    assert result.is_cuda


@pytest.mark.cuda
def test_non_uniform_fast_fourier_op_subspace_gram_cuda() -> None:
    """Test subspace Toeplitz Gram operators work on CUDA devices."""
    rng = RandomGenerator(seed=5)
    n_timepoints, n_coefficients = 4, 2
    image_shape = (12, 10)
    coefficients = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape)).cuda()
    basis = rng.complex64_tensor((n_timepoints, n_coefficients))

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    subspace_gram = nufft_op.toeplitz(subspace=basis).cuda()
    (result,) = subspace_gram(coefficients)
    assert result.is_cuda

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape).cuda()
    subspace_gram = nufft_op.toeplitz(subspace=basis.cuda())
    (result,) = subspace_gram(coefficients)
    assert result.is_cuda


@pytest.mark.cuda
def test_non_uniform_fast_fourier_op_cuda() -> None:
    """Test non-uniform fast Fourier operator works on CUDA devices."""

    # Create a trajectory with non-Cartesian (off-grid) components
    img_shape = (2, 3, 10, 12, 14)
    k_shape = (2, 3, 6, 8, 10)
    nkx = (2, 1, 6, 8, 10)
    nky = (2, 1, 6, 8, 10)
    nkz = (2, 1, 6, 8, 10)
    type_kx = 'non-uniform'
    type_ky = 'non-uniform'
    type_kz = 'non-uniform'

    img, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )

    # Create on CPU, transfer to GPU, run on GPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    operator = nufft_op.H @ nufft_op
    operator.cuda()
    (result,) = operator(img.cuda())
    assert result.is_cuda

    # Create on CPU, run on CPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    operator = nufft_op.H @ nufft_op
    (result,) = operator(img)
    assert result.is_cpu

    # Create on GPU, run on GPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix.cuda(),
        encoding_matrix=encoding_matrix.cuda(),
        traj=trajectory.cuda(),
    )
    operator = nufft_op.H @ nufft_op
    (result,) = operator(img.cuda())
    assert result.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix.cuda(),
        encoding_matrix=encoding_matrix.cuda(),
        traj=trajectory.cuda(),
    )
    operator = nufft_op.H @ nufft_op
    operator.cpu()
    (result,) = operator(img)
    assert result.is_cpu
