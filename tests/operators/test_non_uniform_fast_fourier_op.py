"""Tests for Non-Uniform Fast Fourier operator."""

import pytest
import torch
from mrpro.data import KData, KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import FastFourierOp, NonUniformFastFourierOp

from tests import RandomGenerator
from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.helper import dotproduct_adjointness_test, relative_image_difference


def create_data(im_shape, nkx, nky, nkz, type_kx, type_ky, type_kz) -> tuple[torch.Tensor, KTrajectory]:
    """Create k-space trajectory and random image."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.complex64_tensor(size=im_shape)
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


@COMMON_MR_TRAJECTORIES
def test_non_uniform_fast_fourier_op_fwd_adj_property(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test adjoint property of non-uniform Fast Fourier operator."""

    # generate random images and k-space trajectories
    _, trajectory = create_data(im_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
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
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=im_shape)
    v = random_generator.complex64_tensor(size=k_shape)
    dotproduct_adjointness_test(nufft_op, u, v)


@COMMON_MR_TRAJECTORIES
def test_non_uniform_fast_fourier_op_gram(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test gram of of non-uniform Fast Fourier operator."""
    img, trajectory = create_data(im_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
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

    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=[2, 3, 4, 5])
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
