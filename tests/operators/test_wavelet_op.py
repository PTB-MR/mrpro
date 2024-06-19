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
    """Test transform between ptwt and mrpro coefficient format."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.float32_tensor(size=im_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim)
    if len(dim) == 1:
        coeff_ptwt = wavedec(img, 'haar', level=2, mode='reflect')
        coeff_mrpro = wavelet_op._format_coeffs_1d(coeff_ptwt)
        coeff_ptwt_transformed_1d = wavelet_op._undo_format_coeffs_1d(coeff_mrpro)

        # all entries are single tensors
        for i in range(len(coeff_ptwt_transformed_1d)):
            assert torch.allclose(coeff_ptwt[i], coeff_ptwt_transformed_1d[i])

    elif len(dim) == 2:
        coeff_ptwt = wavedec2(img, 'haar', level=2, mode='reflect')
        coeff_mrpro = wavelet_op._format_coeffs_2d(coeff_ptwt)
        coeff_ptwt_transformed_2d = wavelet_op._undo_format_coeffs_2d(coeff_mrpro)

        # first entry is tensor, the rest is list of tensors
        assert torch.allclose(coeff_ptwt[0], coeff_ptwt_transformed_2d[0])  # type: ignore[arg-type]
        for i in range(1, len(coeff_ptwt_transformed_2d)):
            assert all(
                torch.allclose(coeff_ptwt[i][j], coeff_ptwt_transformed_2d[i][j]) for j in range(len(coeff_ptwt[i]))
            )

    elif len(dim) == 3:
        coeff_ptwt = wavedec3(img, 'haar', level=2, mode='reflect')
        coeff_mrpro = wavelet_op._format_coeffs_3d(coeff_ptwt)
        coeff_ptwt_transformed_3d = wavelet_op._undo_format_coeffs_3d(coeff_mrpro)

        # first entry is tensor, the rest is dict
        assert torch.allclose(coeff_ptwt[0], coeff_ptwt_transformed_3d[0])  # type: ignore[arg-type]
        for i in range(1, len(coeff_ptwt_transformed_3d)):
            assert all(torch.allclose(coeff_ptwt[i][key], coeff_ptwt_transformed_3d[i][key]) for key in coeff_ptwt[i])


def test_wavelet_op_wrong_dim():
    """Wavelet only works for 1D, 2D and 3D data."""
    with pytest.raises(ValueError, match='Only 1D, 2D and 3D wavelet'):
        WaveletOp(dim=(0, 1, 2, 3))  # type: ignore[arg-type]


def test_wavelet_op_mismatch_dim_domain_shape():
    """Dimensions and shapes need to be of same length."""
    with pytest.raises(ValueError, match='Number of dimensions along which'):
        WaveletOp(domain_shape=(10, 20), dim=(-2,))


def test_wavelet_op_error_for_odd_domain_shape():
    with pytest.raises(NotImplementedError, match='ptwt only supports wavelet transforms for tensors with even'):
        WaveletOp(domain_shape=(11, 20), dim=(-2, -1))


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@pytest.mark.parametrize(
    ('im_shape', 'domain_shape', 'dim'),
    [
        ((1, 5, 20, 30), (30,), (-1,)),
        ((5, 1, 10, 20, 30), (10,), (-3,)),
        ((1, 5, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (10, 30), (-3, -1)),
        ((5, 10, 20, 30), (10, 20, 30), (-3, -2, -1)),
        ((6, 10, 20, 30), (6, 20, 30), (-4, -2, -1)),
        ((6, 10, 20, 30), (20, 30, 6), (-2, -1, -4)),
        ((6, 10, 20, 30), (20, 30, 6), (2, 3, 0)),
        ((5, 10, 20, 30), None, (-3, -2, -1)),
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
        ((1, 5, 20, 30), (30,), (-1,)),
        ((5, 1, 10, 20, 30), (10,), (-3,)),
        ((1, 5, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (10, 30), (-3, -1)),
        ((5, 10, 20, 30), (10, 20, 30), (-3, -2, -1)),
        ((6, 10, 20, 30), (6, 20, 30), (-4, -2, -1)),
        ((6, 10, 20, 30), (20, 30, 6), (-2, -1, -4)),
        ((6, 10, 20, 30), (20, 30, 6), (2, 3, 0)),
    ],
)
def test_wavelet_op_adjointness(im_shape, domain_shape, dim, wavelet_name):
    """Test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v."""
    random_generator = RandomGenerator(seed=0)

    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name)

    # calculate 1D length of wavelet coefficients
    wavelet_stack_length = torch.sum(torch.as_tensor([(torch.prod(shape)) for shape in wavelet_op.coefficients_shape]))

    # sorted and normed dimensions needed to correctly calculate range
    dim_sorted = sorted([d % len(im_shape) for d in dim], reverse=True)
    range_shape = list(im_shape)
    range_shape[dim_sorted[-1]] = int(wavelet_stack_length)
    [range_shape.pop(d) for d in dim_sorted[:-1]]

    u = random_generator.complex64_tensor(size=im_shape)
    v = random_generator.complex64_tensor(size=range_shape)
    dotproduct_adjointness_test(wavelet_op, u, v)


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@pytest.mark.parametrize(
    ('im_shape', 'domain_shape', 'dim'),
    [
        ((1, 5, 20, 30), (30,), (-1,)),
        ((5, 1, 10, 20, 30), (10,), (-3,)),
        ((1, 5, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (10, 30), (-3, -1)),
        ((5, 10, 20, 30), (10, 20, 30), (-3, -2, -1)),
        ((6, 10, 20, 30), (6, 20, 30), (-4, -2, -1)),
        ((6, 10, 20, 30), (20, 30, 6), (-2, -1, -4)),
        ((6, 10, 20, 30), (20, 30, 6), (2, 3, 0)),
    ],
)
def test_wavelet_op_unitary(im_shape, domain_shape, dim, wavelet_name):
    """Test if wavelet operator is unitary."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.complex64_tensor(size=im_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name)
    operator_unitary_test(wavelet_op, img)
