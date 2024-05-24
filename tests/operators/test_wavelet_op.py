"""Tests for Wavelet Operator."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from mrpro.operators import WaveletOp
from ptwt.conv_transform import wavedec
from ptwt.conv_transform_2 import wavedec2
from ptwt.conv_transform_3 import wavedec3

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test
from tests.helper import operator_isometry_test
from tests.helper import operator_unitary_test


@pytest.mark.parametrize(
    ('im_shape', 'domain_shape', 'dim'),
    [
        ((5, 16, 16, 16), (16,), (-1,)),
        ((5, 16, 16, 16), (16, 16), (-2, -1)),
        ((5, 16, 16, 16), (16, 16, 16), (-3, -2, -1)),
    ],
)
def test_wavelet_op_coefficient_transform(im_shape, domain_shape, dim):
    random_generator = RandomGenerator(seed=0)
    img = random_generator.float32_tensor(size=im_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim)
    if len(dim) == 1:
        coeff_ptwt = wavedec(img, 'haar', level=2, mode='reflect')
        coeff_mrpro = wavelet_op._format_coeffs_1d(coeff_ptwt)
        coeff_ptwt_transformed = wavelet_op._undo_format_coeffs_1d(coeff_mrpro)
    elif len(dim) == 2:
        coeff_ptwt = wavedec2(img, 'haar', level=2, mode='reflect')
        coeff_mrpro = wavelet_op._format_coeffs_2d(coeff_ptwt)
        coeff_ptwt_transformed = wavelet_op._undo_format_coeffs_2d(coeff_mrpro)
    elif len(dim) == 3:
        coeff_ptwt = wavedec3(img, 'haar', level=2, mode='reflect')
        coeff_mrpro = wavelet_op._format_coeffs_3d(coeff_ptwt)
        coeff_ptwt_transformed = wavelet_op._undo_format_coeffs_3d(coeff_mrpro)

    for i in range(len(coeff_ptwt)):
        if isinstance(coeff_ptwt[i], dict):
            assert all(torch.allclose(coeff_ptwt[i][key], coeff_ptwt_transformed[i][key]) for key in coeff_ptwt[i])
        elif isinstance(coeff_ptwt[i], torch.Tensor):
            assert torch.allclose(coeff_ptwt[i], coeff_ptwt_transformed[i])
        elif isinstance(coeff_ptwt[i], tuple):
            assert all(
                torch.allclose(coeff_ptwt[i][j], coeff_ptwt_transformed[i][j]) for j in range(len(coeff_ptwt[i]))
            )


def test_wavelet_op_wrong_dim():
    with pytest.raises(ValueError, match='Only 1D, 2D and 3D wavelet'):
        WaveletOp(dim=(0, 1, 2, 3))


def test_wavelet_op_mismatch_dim_domain_shape():
    with pytest.raises(ValueError, match='Number of dimensions along which'):
        WaveletOp(domain_shape=(10, 20), dim=(-2,))


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@pytest.mark.parametrize(
    ('im_shape', 'domain_shape', 'dim'),
    [
        ((1, 32, 32), (32,), (-1,)),
        ((1, 32, 32), (32, 32), (-2, -1)),
        ((4, 30, 30), (30, 30), (-2, -1)),
        ((5, 1, 32, 32, 32), (32, 32, 32), (-3, -2, -1)),
        ((5, 1, 30, 20, 40), (30,), (-3,)),
    ],
)
def test_wavelet_op_isometry(im_shape, domain_shape, dim, wavelet_name):
    """Test that the wavelet operator is a linear isometry."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.complex64_tensor(size=im_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name, level=None)
    operator_isometry_test(wavelet_op, img)


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@pytest.mark.parametrize(
    ('im_shape', 'domain_shape', 'dim'),
    [
        ((1, 32, 32), (32,), (-1,)),
        ((1, 32, 32), (32, 32), (-2, -1)),
        ((4, 30, 30), (30, 30), (-2, -1)),
        ((5, 1, 16, 32, 32), (16, 32, 32), (-3, -2, -1)),
        ((5, 1, 30, 20, 40), (30,), (-3,)),
    ],
)
def test_wavelet_op_adjointness(im_shape, domain_shape, dim, wavelet_name):
    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)

    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name)
    # calculate 1D length of wavelet coefficients
    range_shape = [(torch.prod(shape)) for shape in wavelet_op.coefficients_shape]

    u = random_generator.complex64_tensor(size=im_shape)
    v = random_generator.complex64_tensor(size=im_shape[: -len(dim)] + (torch.sum(torch.as_tensor(range_shape)),))
    dotproduct_adjointness_test(wavelet_op, u, v)


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@pytest.mark.parametrize(
    ('im_shape', 'domain_shape', 'dim'),
    [
        ((1, 32, 32), (32,), (-1,)),
        ((1, 32, 32), (32, 32), (-2, -1)),
        ((4, 30, 30), (30, 30), (-2, -1)),
        ((5, 1, 16, 32, 32), (16, 32, 32), (-3, -2, -1)),
        ((5, 1, 30, 20, 40), (30,), (-3,)),
    ],
)
def test_wavelet_op_unitary(im_shape, domain_shape, dim, wavelet_name):
    """Test if wavelet operator is unitary."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.complex64_tensor(size=im_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name)
    operator_unitary_test(wavelet_op, img)
