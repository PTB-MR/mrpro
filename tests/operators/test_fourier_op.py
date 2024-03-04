"""Tests for Fourier operator."""

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
from mrpro.data import SpatialDimension
from mrpro.operators import FourierOp

from tests import RandomGenerator
from tests.conftest import COMMON_MR_TRAJECTORIES
from tests.data.test_ktraj import create_traj


def create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    image = random_generator.complex64_tensor(size=im_shape)
    # create random trajectories
    ktraj = create_traj(k_shape, nkx, nky, nkz, sx, sy, sz)
    return image, ktraj


@COMMON_MR_TRAJECTORIES
def test_fourier_fwd_adj_property(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    """Test adjoint property of Fourier operator."""

    # generate random images and k-space trajectories
    image, ktraj = create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz)

    # create operator
    recon_shape = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_shape = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    op = FourierOp(recon_shape=recon_shape, encoding_shape=encoding_shape, traj=ktraj)

    # apply forward and adjoint operator
    (kdata,) = op(image)
    (reco,) = op.H(kdata)

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=image.shape)
    v = random_generator.complex64_tensor(size=kdata.shape)
    (Fu,) = op(u)
    (FHv,) = op.H(v)
    Fu_v = torch.vdot(Fu.flatten(), v.flatten())
    u_FHv = torch.vdot(u.flatten(), FHv.flatten())

    # Check that the dimensions are correct
    assert reco.shape == image.shape

    # Check the adjoint property
    assert torch.isclose(Fu_v, u_FHv, rtol=1e-3)


@pytest.mark.parametrize(
    ('im_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'sx', 'sy', 'sz', 's0', 's1', 's2'),
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
    image, ktraj = create_data(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz)

    # create operator
    recon_shape = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_shape = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    with pytest.raises(NotImplementedError, match='Cartesian FFT dims need to be aligned'):
        FourierOp(recon_shape=recon_shape, encoding_shape=encoding_shape, traj=ktraj)
