"""Tests for finite difference operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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
from mrpro.operators import FiniteDifferenceOp

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
def test_finite_difference_op_forward(mode):
    """Test correct finite difference of simple object."""
    # Test object with linear gradient in real and imaginary part
    linear_gradient_object = torch.arange(1, 21)[None, :].to(dtype=torch.float32)
    linear_gradient_object = linear_gradient_object + 1j * linear_gradient_object

    # Generate and apply finite difference operator
    finite_difference_op = FiniteDifferenceOp(dim=(-1,), mode=mode)
    (finite_difference_of_object,) = finite_difference_op(linear_gradient_object)

    # Verify correct values excluding borders
    torch.testing.assert_close(finite_difference_of_object[0, 0, 1:-1], (1 + 1j) * torch.ones(18))


@pytest.mark.parametrize('padding_mode', ['zero', 'reflect', 'replicate', 'circular'])
@pytest.mark.parametrize('mode', ['central', 'forward', 'backward'])
@pytest.mark.parametrize('dim', [(-1,), (-2, -1), (-3, -2, -1), (-4,), (1, 3)])
def test_finite_difference_op_adjointness(dim, mode, padding_mode):
    """Test finite difference operator adjoint property."""

    random_generator = RandomGenerator(seed=0)
    im_shape = (5, 6, 4, 10, 20, 16)

    # Generate finite difference operator
    finite_difference_op = FiniteDifferenceOp(dim, mode, padding_mode)

    # Check adjoint property
    u = random_generator.complex64_tensor(size=im_shape)
    v = random_generator.complex64_tensor(size=(len(dim), *im_shape))
    dotproduct_adjointness_test(finite_difference_op, u, v)
