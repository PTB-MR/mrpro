"""Tests for Non-Uniform Fast Fourier operator."""

import pytest
import torch
from mrpro.data import KData, KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import FastFourierOp, NonUniformFastFourierOp
from mrpro.utils import RandomGenerator

from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.helper import dotproduct_adjointness_test, relative_image_difference


def create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz) -> tuple[torch.Tensor, KTrajectory]:
    """Create k-space trajectory and random image."""
    rng = RandomGenerator(seed=0)
    img = rng.complex64_tensor(size=img_shape)
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


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
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_yx(img)[0])


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
