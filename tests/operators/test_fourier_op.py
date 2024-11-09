"""Tests for Fourier operator."""

import pytest
from mrpro.data import SpatialDimension
from mrpro.operators import FourierOp

from tests import RandomGenerator
from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.helper import dotproduct_adjointness_test


def create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    img = random_generator.complex64_tensor(size=im_shape)
    # create random trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


@COMMON_MR_TRAJECTORIES
def test_fourier_fwd_adj_property(
    im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
):
    """Test adjoint property of Fourier operator."""

    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    # apply forward operator
    (kdata,) = fourier_op(img)

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=img.shape)
    v = random_generator.complex64_tensor(size=kdata.shape)
    dotproduct_adjointness_test(fourier_op, u, v)


@pytest.mark.parametrize(
    ('im_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'type_kx', 'type_ky', 'type_kz'),  # parameter names
    [
        (  # Cartesian FFT dimensions are not aligned with corresponding k2, k1, k0 dimensions
            (5, 3, 48, 16, 32),  # im_shape
            (5, 3, 96, 18, 64),  # k_shape
            (5, 1, 18, 64),  # nkx
            (5, 96, 1, 1),  # nky - Cartesian ky dimension defined along k2 rather than k1
            (5, 1, 18, 64),  # nkz
            'non-uniform',  # type_kx
            'uniform',  # type_ky
            'non-uniform',  # type_kz
        ),
    ],
    ids=['cartesian_fft_dims_not_aligned_with_k2_k1_k0_dims'],
)
def test_fourier_not_supported_traj(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz):
    """Test trajectory not supported by Fourier operator."""

    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    with pytest.raises(NotImplementedError, match='Cartesian FFT dims need to be aligned'):
        FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)
