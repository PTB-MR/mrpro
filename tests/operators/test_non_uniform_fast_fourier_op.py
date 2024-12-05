"""Tests for Non-Uniform Fast Fourier operator."""

import pytest
import torch
from mrpro.data import KData, KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.operators import NonUniformFastFourierOp

from tests import RandomGenerator
from tests.conftest import create_traj
from tests.helper import dotproduct_adjointness_test, relative_image_difference


def create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    img = random_generator.complex64_tensor(size=im_shape)
    # create random trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


@pytest.mark.parametrize('dim', [(-1,), (-1, -2), (-1, -2, -3), (-2, -3), (-3, -1)])
def test_non_uniform_fast_fourier_op_fwd_adj_property(dim):
    """Test adjoint property of non-uniform fast Fourier operator."""

    kdata_shape = [2, 3, 20, 30, 40]
    recon_matrix = [20, 20, 30]
    img_shape = kdata_shape[:2] + [recon_matrix[d] if d in dim else kdata_shape[d] for d in (-3, -2, -1)]

    # generate random traj
    nk = [kdata_shape[d] if d in dim else 1 for d in (-5, -3, -2, -1)]  # skip coil dimension
    traj = create_traj(
        kdata_shape, nkx=nk, nky=nk, nkz=nk, type_kx='non-uniform', type_ky='non-uniform', type_kz='non-uniform'
    )

    # create operator
    nufft_op = NonUniformFastFourierOp(
        direction=dim,
        recon_matrix=[recon_matrix[d] for d in dim],
        encoding_matrix=[kdata_shape[d] for d in dim],
        traj=traj,
    )

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=img_shape)
    v = random_generator.complex64_tensor(size=kdata_shape)
    dotproduct_adjointness_test(nufft_op, u, v)


def test_non_uniform_fast_fourier_op_equal_to_fft(ismrmrd_cart):
    """Eval result of non-uniform fourier transform for Cartesian data."""
    kdata = KData.from_file(ismrmrd_cart.filename, KTrajectoryIsmrmrd())

    # recon_matrix and encoding_matrix have to be identical to avoid image scaling
    # oversampling > 1 leads to a scaling of the image, the object of the images are far away from the edge so there
    # are no aliasing artifacts even for oversampling = 1
    nufft_op = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
        encoding_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
        traj=kdata.traj,
        nufft_oversampling=1.0,
    )
    (reconstructed_img,) = nufft_op.adjoint(kdata.data)

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img[0, 0, 0, ...], ismrmrd_cart.img_ref) <= 0.05


def test_non_uniform_fast_fourier_op_empty_dims():
    """Empty dims do not change the input."""
    nk = [1, 1, 1, 1, 1]
    traj = create_traj(nk, nkx=nk, nky=nk, nkz=nk, type_kx='non-uniform', type_ky='non-uniform', type_kz='non-uniform')
    nufft_op = NonUniformFastFourierOp(direction=(), recon_matrix=(), encoding_matrix=(), traj=traj)

    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=[2, 3, 4, 5])
    torch.testing.assert_close(u, nufft_op(u)[0])


def test_non_uniform_fast_fourier_op_not_aligned_dim_directions():
    """Test case where directions and k-space dimensions are not alined."""

    kdata_shape = [1, 3, 20, 30, 40]
    img_shape = [2, 3, 20, 20, 30]
    nkx = [1, 1, 30, 40]
    nky = [1, 20, 1, 1]  # Cartesian ky dimension defined along k2 rather than k1
    nkz = [1, 1, 30, 40]

    # generate random traj and image
    img, traj = create_data(
        img_shape,
        kdata_shape,
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        type_kx='non-uniform',
        type_ky='uniform',
        type_kz='non-uniform',
    )

    # create operator
    nufft_op = NonUniformFastFourierOp(
        direction=('x', 'z'),
        recon_matrix=SpatialDimension[int](z=img_shape[-3], y=img_shape[-2], x=img_shape[-1]),
        encoding_matrix=SpatialDimension[int](z=kdata_shape[-3], y=kdata_shape[-2], x=kdata_shape[-1]),
        traj=traj,
    )

    assert nufft_op(img)


def test_non_uniform_fast_fourier_op_directions():
    """Test different direction specifiers of non-uniform fast Fourier operator."""

    kdata_shape = [2, 3, 20, 30, 40]
    recon_matrix = [20, 20, 30]
    img_shape = [2, 3, 20, 20, 30]

    # generate random traj and image
    nk = [2, 20, 30, 40]
    img, traj = create_data(
        img_shape,
        kdata_shape,
        nkx=nk,
        nky=nk,
        nkz=nk,
        type_kx='non-uniform',
        type_ky='non-uniform',
        type_kz='non-uniform',
    )

    # create operator
    dim = [-2, -1]
    nufft_op_12 = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=[recon_matrix[d] for d in dim],
        encoding_matrix=[kdata_shape[d] for d in dim],
        traj=traj,
    )

    nufft_op_yx = NonUniformFastFourierOp(
        direction=('y', 'x'),
        recon_matrix=[recon_matrix[d] for d in dim],
        encoding_matrix=[kdata_shape[d] for d in dim],
        traj=traj,
    )
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_yx(img)[0])


def test_non_uniform_fast_fourier_op_error_directions():
    """Test error for duplicate directions of non-uniform fast Fourier operator."""
    with pytest.raises(ValueError, match='Directions must be unique'):
        NonUniformFastFourierOp(
            direction=(-2, -1, 'x'),
            recon_matrix=[1, 1],
            encoding_matrix=[1, 1],
            traj=KTrajectory.from_tensor(torch.randn((3, 1, 1, 1, 1))),
        )


def test_non_uniform_fast_fourier_op_error_matrix():
    """Test error for wrong matrix dimensions of non-uniform fast Fourier operator."""
    with pytest.raises(ValueError, match='recon_matrix should have'):
        NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=[1, 1, 1],
            encoding_matrix=[1, 1],
            traj=KTrajectory.from_tensor(torch.randn((3, 1, 1, 1, 1))),
        )

    with pytest.raises(ValueError, match='encoding_matrix should have'):
        NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=[1, 1],
            encoding_matrix=[1],
            traj=KTrajectory.from_tensor(torch.randn((3, 1, 1, 1, 1))),
        )
