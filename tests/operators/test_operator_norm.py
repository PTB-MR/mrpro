"""Tests for computing the operator norm of linear operators."""

from math import prod, sqrt

import pytest
import torch
from mrpro.operators import EinsumOp, FastFourierOp, FiniteDifferenceOp

from tests import RandomGenerator


def test_power_iteration_uses_stopping_criterion():
    """Test if the power iteration stops if the absolute and relative tolerance are chosen high."""

    def callback(_):
        """Callback function that should not be called, because the power iteration should stop."""
        pytest.fail('The power iteration did not stop despite high atol and rtol!')

    random_generator = RandomGenerator(seed=0)

    # test with a 3x3 matrix
    matrix = torch.tensor([[2.0, 1, 0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    operator = EinsumOp(matrix, ' y x, x-> y')
    random_vector = random_generator.float32_tensor(matrix.shape[1])
    absolute_tolerance, relative_tolerance = 1e8, 1e8
    _ = operator.operator_norm(
        random_vector,
        dim=None,
        max_iterations=32,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        callback=callback,
    )


def test_operator_norm_invalid_max_iterations():
    """Test if choosing a number of iterations < 1 throws an exception."""
    random_generator = RandomGenerator(seed=0)

    # test with a 3x3 matrix with known largest eigenvalue
    matrix = torch.tensor([[2.0, 1, 0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    operator = EinsumOp(matrix, 'y x, x -> y')
    random_vector = random_generator.float32_tensor(matrix.shape[1])

    with pytest.raises(ValueError, match='zero'):
        operator.operator_norm(random_vector, dim=None, max_iterations=0)


def test_operator_norm_invalid_initial_value():
    """Test if choosing zero-tensors throws an exception."""
    random_generator = RandomGenerator(seed=0)
    input_shape = (2, 4, 8, 8)
    vector_shape = (input_shape[0], input_shape[1], input_shape[3])

    # create a tensor to be identified as 2 * 4 (=8) 8x8 square matrices
    matrix = random_generator.float32_tensor(size=input_shape)

    # construct a linear operator from the first matrix; the linear operator implements
    # the batched matrix-vector multiplication
    operator = EinsumOp(matrix, '... y x, ... x-> ... y')

    # dimensions which define the dimensionality of the considered vector space
    dim1 = (-1,)
    dim2 = None

    # random vector with only one of the sub-vector being a zero-vector;
    illegal_initial_value1 = random_generator.float32_tensor(size=vector_shape)
    illegal_initial_value1[0] = 0.0

    # zero-vector
    illegal_initial_value2 = torch.zeros(vector_shape)

    with pytest.raises(ValueError, match='least'):
        operator.operator_norm(illegal_initial_value1, dim=dim1, max_iterations=8)
    with pytest.raises(ValueError, match='zero'):
        operator.operator_norm(illegal_initial_value2, dim=dim2, max_iterations=8)


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
    """Test with Fast Fourier Operator (has norm 1 since norm="ortho" is used)."""
    random_generator = RandomGenerator(seed=0)

    dim = (-3, -2, -1)
    fourier_op = FastFourierOp(dim=dim)
    random_image = random_generator.complex64_tensor(size=(4, 4, 8, 16))
    fourier_op_norm_batched = fourier_op.operator_norm(random_image, dim=dim, max_iterations=128)
    fourier_op_norm_non_batched = fourier_op.operator_norm(random_image, dim=None, max_iterations=128)
    fourier_op_norm_true = 1.0

    torch.testing.assert_close(fourier_op_norm_batched, torch.ones(4, 1, 1, 1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fourier_op_norm_non_batched.max().item(), fourier_op_norm_true, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fourier_op_norm_non_batched.item(), fourier_op_norm_true, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    'dim',
    [(-1,), (-2, -1), (-3, -2, -1), (-4, -3, -2, -1)],
)
def test_finite_difference_operator_norm(dim):
    """Test with the finite difference operator for which there exists a closed-form solution."""
    random_generator = RandomGenerator(seed=0)

    finite_difference_operator = FiniteDifferenceOp(dim=dim, mode='forward')

    # initialize random image of appropriate shape depending on the dimensionality
    image_shape = (1, *tuple([16 for _ in range(len(dim))]))
    random_image = random_generator.complex64_tensor(size=image_shape)

    # calculate the operator norm
    finite_difference_operator_norm = finite_difference_operator.operator_norm(random_image, dim=dim, max_iterations=64)

    # closed form solution of the operator norm
    finite_difference_operator_norm_true = sqrt(len(dim) * 4)

    torch.testing.assert_close(
        finite_difference_operator_norm.item(), finite_difference_operator_norm_true, atol=1e-2, rtol=1e-2
    )


def test_batched_operator_norm():
    """Test if the batched calculation of the operator-norm works on a simple
    matrix-vector multiplication example.

    Using the fact that for a block-diagonal matrix, the eigenvalues are the list of
    eigenvalues of the respective matrices, we test whether the largest of the batched
    operator norms is equal to the non-batched operator norm.
    """
    random_generator = RandomGenerator(seed=0)
    input_shape = (2, 4, 8, 8)

    # dimensions which define the dimensionality of the considered vector space
    dim = (-1,)

    # create a tensor to be identified as 2 * 4 (=8) 8x8 square matrices
    matrix1 = torch.arange(prod(input_shape)).reshape(input_shape).to(torch.float32)

    # construct a linear operator from the first matrix; the linear operator implements
    # the batched matrix-vector multiplication
    operator1 = EinsumOp(matrix1, '... y x, ... x-> ... y')

    random_vector1 = random_generator.float32_tensor((input_shape[0], input_shape[1], input_shape[3]))

    # compute batched and non batched operator norms
    operator1_norm_batched = operator1.operator_norm(random_vector1, dim=dim, max_iterations=32)
    operator1_norm_non_batched = operator1.operator_norm(random_vector1, dim=None, max_iterations=32)

    torch.testing.assert_close(
        operator1_norm_batched.max().item(), operator1_norm_non_batched.item(), atol=1e-4, rtol=1e-4
    )

    # create a block diagonal matrix containing the 2*4=8 matrices in the diagonal
    matrix2 = torch.block_diag(
        *[matrix1[other0, other1, ...] for other0 in range(input_shape[0]) for other1 in range(input_shape[1])]
    )

    # construct a linear operator from the second matrix; the linear operator implements
    # the multiplication of the block-diagonal matrix with a 2*4*8*8 = 512-dimensional vector
    operator2 = EinsumOp(matrix2, '... y x, x-> ... y')
    random_vector2 = random_generator.float32_tensor(matrix2.shape[1])
    operator2_norm_non_batched = operator2.operator_norm(random_vector2, dim=None, max_iterations=32)

    # test whether the operator-norm calculated from the first operator and from the second
    # are equal
    torch.testing.assert_close(
        operator2_norm_non_batched.item(), operator1_norm_non_batched.item(), atol=1e-4, rtol=1e-4
    )
