"""Tests for Fourier operator."""

import pytest
from mrpro.data import SpatialDimension
from mrpro.operators import FourierOp

from tests import RandomGenerator
from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.helper import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    img = random_generator.complex64_tensor(size=im_shape)
    # create random trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)
    return img, trajectory


def create_fourier_op_and_range_domain(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    """Create a fourier operator and an element from domain and range."""
    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz)

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
    return fourier_op, u, v


@COMMON_MR_TRAJECTORIES
def test_fourier_fwd_adj_property(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz, s0, s1, s2):
    """Test adjoint property of Fourier operator."""
    dotproduct_adjointness_test(*create_fourier_op_and_range_domain(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz))


def test_finite_difference_op_grad():
    """Test gradient of fourier operator."""
    im_shape = (2, 8, 64, 32, 48)
    k_shape = (2, 8, 8, 64, 96)
    nkx = (2, 1, 1, 96)
    nky = (2, 8, 64, 1)
    nkz = (2, 8, 64, 1)
    gradient_of_linear_operator_test(
        *create_fourier_op_and_range_domain(im_shape, k_shape, nkx, nky, nkz, 'uf', 'nuf', 'nuf')
    )


def test_finite_difference_op_forward_mode_autodiff():
    """Test forward-mode autodiff of fourier operator."""
    im_shape = (2, 8, 64, 32, 48)
    k_shape = (2, 8, 8, 64, 96)
    nkx = (2, 1, 1, 96)
    nky = (2, 8, 64, 1)
    nkz = (2, 8, 64, 1)
    forward_mode_autodiff_of_linear_operator_test(
        *create_fourier_op_and_range_domain(im_shape, k_shape, nkx, nky, nkz, 'uf', 'nuf', 'nuf')
    )


@pytest.mark.parametrize(
    ('im_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'sx', 'sy', 'sz'),
    [
        # Cartesian FFT dimensions are not aligned with corresponding k2, k1, k0 dimensions
        (
            (5, 3, 48, 16, 32),
            (5, 3, 96, 18, 64),
            (5, 1, 18, 64),
            (5, 96, 1, 1),  # Cartesian ky dimension defined along k2 rather than k1
            (5, 1, 18, 64),
            'nuf',
            'uf',
            'nuf',
        ),
    ],
)
def test_fourier_not_supported_traj(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    """Test trajectory not supported by Fourier operator."""

    # generate random images and k-space trajectories
    img, trajectory = create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz)

    # create operator
    recon_matrix = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    with pytest.raises(NotImplementedError, match='Cartesian FFT dims need to be aligned'):
        FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)
