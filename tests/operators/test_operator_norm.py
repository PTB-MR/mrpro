"""Tests for computing the operator norm of linear operators."""

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
from math import prod
from math import sqrt

import torch
from mrpro.operators import EinsumOp
from mrpro.operators import FastFourierOp

from tests import RandomGenerator


def test_operator_norm_result():
    """Test if the implementation yields the correct result for different choices
    of operators with known operator-norm."""
    random_generator = RandomGenerator(seed=0)

    # test with a 3x3 matrix with known largest eigenvalue
    matrix = torch.tensor([[2.0, 1, 0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    operator = EinsumOp(matrix, 'y x, x-> y')
    random_vector = random_generator.float32_tensor(matrix.shape[1])
    operator_norm_est = operator.operator_norm(random_vector, dim=None, max_iterations=32)
    operator_norm_true = 2 + sqrt(2)  # approximately 3.41421...
    torch.testing.assert_close(operator_norm_est.item(), operator_norm_true, atol=1e-4, rtol=1e-4)


def test_fourier_operator_norm():
    """Test with Fast Fourier Operator (has norm 1 since norm="ortho" is used);
    # also tests that the initial value is set to a random value if chosen as zero."""
    dim = (-3, -2, -1)
    fourier_op = FastFourierOp(dim=dim)

    random_image = torch.zeros(4, 4, 8, 16, dtype=torch.complex64)
    fourier_op_norm_batched = fourier_op.operator_norm(random_image, dim=dim, max_iterations=64)
    fourier_op_norm_non_batched = fourier_op.operator_norm(random_image, dim=None, max_iterations=64)
    fourier_op_norm_true = 1.0

    torch.testing.assert_close(fourier_op_norm_batched, torch.ones(4, 1, 1, 1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fourier_op_norm_non_batched.max().item(), fourier_op_norm_true, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fourier_op_norm_non_batched.item(), fourier_op_norm_true, atol=1e-4, rtol=1e-4)


def test_batched_operator_norm():
    """Test if the batched calculation of the operator-norm works on a simple
    matrix-vector multiplication example.

    Using the fact that for a block-diagonal matrix, the eigenvalues are the list of
    eigenvalues of the respective matrices, we test whether the largest of the batched
    operator norms coincides with the non-batched operator norm.
    """
    random_generator = RandomGenerator(seed=0)
    input_shape = (2, 4, 8, 8)

    # dimensions which define the dimensionality of the considered vector space
    dim = (-1,)

    # create a tensor to be identified as 2 * 4 (=8) 8x8 square matrices
    matrix1 = torch.arange(prod(input_shape)).reshape(input_shape).to(torch.float32)

    # construct a linear operator from the first matrix; the linear operator implements
    # the batched matrix-vector multiplication
    operator1 = EinsumOp(matrix1, 'other1 other2 y x, other1 other2 x-> other1 other2 y')

    random_vector1 = random_generator.float32_tensor((input_shape[0], input_shape[1], input_shape[3]))

    # compute batched and non batched operator norms
    operator1_norm_batched = operator1.operator_norm(random_vector1, dim=dim, max_iterations=32)
    operator1_norm_non_batched = operator1.operator_norm(random_vector1, dim=None, max_iterations=32)

    torch.testing.assert_close(
        operator1_norm_batched.max().item(), operator1_norm_non_batched.item(), atol=1e-4, rtol=1e-4
    )

    # create a block diagonal matrix containing the 2*4=8 matrices in the diagonal
    matrix2 = torch.block_diag(*[matrix1[kb, kt, ...] for kb in range(input_shape[0]) for kt in range(input_shape[1])])

    # construct a linear operator from the second matrix; the linear operator implements
    # the multiplication of the block-diagonal matrix with a 2*4*8*8 = 512-dimensional vector
    operator2 = EinsumOp(matrix2, 'y x, x-> y')
    random_vector2 = random_generator.float32_tensor(matrix2.shape[1])
    operator2_norm_non_batched = operator2.operator_norm(random_vector2, dim=None, max_iterations=32)

    # test whether the operator-norm calculated from the first operator and from the second
    # one coincide
    torch.testing.assert_close(
        operator2_norm_non_batched.item(), operator1_norm_non_batched.item(), atol=1e-4, rtol=1e-4
    )
