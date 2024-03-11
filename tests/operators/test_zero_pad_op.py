"""Tests for Zero Pad Operator class."""

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
from mrpro.operators import ZeroPadOp

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def test_zero_pad_op_content():
    """Test correct padding and cropping (i.e. negative padding size)."""
    original_shape = (2, 100, 3, 200, 50, 2)
    padded_shape = (2, 80, 3, 100, 240, 2)
    generator = RandomGenerator(seed=0)
    original_data = generator.complex64_tensor(original_shape)
    padding_dimensions = (-5, -3, -2)
    zero_padding_op = ZeroPadOp(
        dim=padding_dimensions,
        original_shape=tuple([original_shape[d] for d in padding_dimensions]),
        padded_shape=tuple([padded_shape[d] for d in padding_dimensions]),
    )
    (padded_data,) = zero_padding_op.forward(original_data)

    # Compare overlapping region
    torch.testing.assert_close(original_data[:, 10:90, :, 50:150, :, :], padded_data[:, :, :, :, 95:145, :])


@pytest.mark.parametrize(
    ('u_shape', 'v_shape'),
    [
        ((101, 201, 50), (13, 221, 64)),
        ((100, 200, 50), (14, 220, 64)),
        ((101, 201, 50), (14, 220, 64)),
        ((100, 200, 50), (13, 221, 64)),
    ],
)
def test_zero_pad_op_adjoint(u_shape, v_shape):
    """Test adjointness of pad operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(u_shape)
    v = generator.complex64_tensor(v_shape)
    zero_padding_op = ZeroPadOp(dim=(-3, -2, -1), original_shape=u_shape, padded_shape=v_shape)
    dotproduct_adjointness_test(zero_padding_op, u, v)
