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
from mrpro.operators import WaveletOp

from tests import RandomGenerator


def test_wavelet_op_wrong_dim():
    with pytest.raises(ValueError, match='Only 1D, 2D and 3D wavelet'):
        WaveletOp(dim=(0, 1, 2, 3))


def test_wavelet_op_mismatch_dim_domain_shape():
    with pytest.raises(ValueError, match='Number of dimensions along which'):
        WaveletOp(domain_shape=(10, 20), dim=(-2,))


@pytest.mark.parametrize(
    ('im_shape', 'domain_shape', 'dim'),
    [
        ((10, 20, 30), (30,), (-1,)),
        ((10, 20, 30), (20, 30), (-2, -1)),
        ((10, 20, 30), (10, 20, 30), (-3, -2, -1)),  # error because default to level=0
        ((10, 20, 30), (10, 20), (-3, -2)),  # error because default to level=0
    ],
)
def test_wavelet_op_isometry(im_shape, domain_shape, dim):
    random_generator = RandomGenerator(seed=0)
    img = random_generator.float32_tensor(size=im_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim)
    wavelet_coefficients = wavelet_op(img)
    assert wavelet_coefficients is not None


def test_wavelet_op_adjointness():
    pass


def test_wavelet_op_unitary():
    pass
