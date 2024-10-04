import sys

import pytest
import torch

sys.path.append('/home/zimmer08/code/mrpro/')
from mrpro.operators import EinsumOp
from mrpro.operators.LinearOperatorMatrix import LinearOperatorMatrix

from tests import RandomGenerator


def random_linearop(size, rng):
    return EinsumOp(rng.complex64_tensor(size), '... i j, ... j -> ... i')


def random_linearoperatormatrix(size, inner_size, rng):
    operators = [[random_linearop(inner_size, rng) for i in range(size[1])] for j in range(size[0])]
    return LinearOperatorMatrix(operators)


def test_linearoperatormatrix_shape():
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((5, 3), (3, 10), rng)
    assert matrix.shape == (5, 3)


def test_linearoperatormatrix_add_matrix():
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    matrix2 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 + matrix2)(*vector)
    expected = tuple(a + b for a, b in zip(matrix1(*vector), matrix2(*vector), strict=False))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_add_tensor_nonsquare():
    """Add tensor to non-square matrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(3)
    with pytest.raises(NotImplementedError, match='square'):
        (matrix1 + other)


def test_linearoperatormatrix_add_tensor_square():
    """Add tensor to square matrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 5), (3, 10), rng)
    other = rng.complex64_tensor(5)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 + other)(vector)
    expected = tuple((mv + o * v for mv, o, v in zip(matrix1(*vector), other, vector, strict=False)))

    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_rmul():
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(3)
    vector = rng.complex64_tensor((3, 10))
    result = (other * matrix1)(*vector)
    expected = tuple(other * el for el in matrix1(*vector))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_mul():
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(10)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 * other)(*vector)
    expected = matrix1(*(other * el for el in vector))
    torch.testing.assert_close(result, expected)
