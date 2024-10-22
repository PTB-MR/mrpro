"""Tests for Non-Uniform Fast Fourier operator."""

import pytest
from mrpro.operators import NonUniformFastFourierOp

from tests import RandomGenerator
from tests.conftest import create_traj
from tests.helper import dotproduct_adjointness_test


def create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    img = random_generator.complex64_tensor(size=im_shape)
    # create random trajectories
    trajectory = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)
    return img, trajectory


@pytest.mark.parametrize('dim', [(-1,), (-1, -2), (-1, -2, -3), (-2, -3), (-3, -1)])
def test_fourier_fwd_adj_property(dim):
    """Test adjoint property of Fourier operator."""

    kdata_shape = [2, 3, 20, 30, 40]
    recon_matrix = [20, 20, 30]
    img_shape = kdata_shape[:2] + [recon_matrix[d] if d in dim else kdata_shape[d] for d in (-3, -2, -1)]

    # generate random traj
    nk = [kdata_shape[d] if d in dim else 1 for d in (-5, -3, -2, -1)]  # skip coil dimension
    traj = create_traj(kdata_shape, nkx=nk, nky=nk, nkz=nk, sx='nuf', sy='nuf', sz='nuf')

    # create operator
    nufft_op = NonUniformFastFourierOp(
        dim=dim,
        recon_matrix=[recon_matrix[d] for d in dim],
        encoding_matrix=[kdata_shape[d] for d in dim],
        traj=traj,
    )

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=img_shape)
    v = random_generator.complex64_tensor(size=kdata_shape)
    dotproduct_adjointness_test(nufft_op, u, v)
