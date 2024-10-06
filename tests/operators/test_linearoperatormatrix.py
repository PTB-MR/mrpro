import pytest
import torch
from mrpro.operators import EinsumOp, LinearOperator
from mrpro.operators.LinearOperatorMatrix import LinearOperatorMatrix

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


def random_linearop(size, rng):
    """Create a random LinearOperator."""
    return EinsumOp(rng.complex64_tensor(size), '... i j, ... j -> ... i')


def random_linearoperatormatrix(size, inner_size, rng):
    """Create a random LinearOperatorMatrix."""
    operators = [[random_linearop(inner_size, rng) for i in range(size[1])] for j in range(size[0])]
    return LinearOperatorMatrix(operators)


def test_linearoperatormatrix_shape():
    """Test creation and shape of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((5, 3), (3, 10), rng)
    assert matrix.shape == (5, 3)


def test_linearoperatormatrix_add_matrix():
    """Test addition of two LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    matrix2 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 + matrix2)(*vector)
    expected = tuple(a + b for a, b in zip(matrix1(*vector), matrix2(*vector), strict=False))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_add_tensor_nonsquare():
    """Test failure of addition of tensor to non-square matrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(3)
    with pytest.raises(NotImplementedError, match='square'):
        (matrix1 + other)


def test_linearoperatormatrix_add_tensor_square():
    """Add tensor to square matrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((3, 3), (2, 2), rng)
    other = rng.complex64_tensor(2)
    vector = rng.complex64_tensor((3, 2))
    result = (matrix1 + other)(*vector)
    expected = tuple((mv + other * v for mv, v in zip(matrix1(*vector), vector, strict=True)))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_rmul():
    """Test post multiplication with tensor."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(3)
    vector = rng.complex64_tensor((3, 10))
    result = (other * matrix1)(*vector)
    expected = tuple(other * el for el in matrix1(*vector))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_mul():
    """Test pre multiplication with tensor."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(10)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 * other)(*vector)
    expected = matrix1(*(other * el for el in vector))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_composition():
    """Test composition of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((1, 5), (2, 3), rng)
    matrix2 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 @ matrix2)(*vector)
    expected = matrix1(*(matrix2(*vector)))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_composition_mismatch():
    """Test composition with mismatching shapes."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((1, 5), (2, 3), rng)
    matrix2 = random_linearoperatormatrix((4, 3), (3, 10), rng)
    vector = rng.complex64_tensor((4, 10))
    with pytest.raises(ValueError, match='shapes do not match'):
        (matrix1 @ matrix2)(*vector)


def test_linearoperatormatrix_adjoint():
    """Test adjointness of Adjoint."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((5, 3), (3, 10), rng)

    class Wrapper(LinearOperator):
        """Stack the output of the matrix operator."""

        def forward(self, x):
            return (torch.stack(matrix(*x), 0),)

        def adjoint(self, x):
            return (torch.stack(matrix.adjoint(*x), 0),)

    dotproduct_adjointness_test(Wrapper(), rng.complex64_tensor((3, 10)), rng.complex64_tensor((5, 3)))


def test_linearoperatormatrix_repr():
    """Test repr of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((5, 3), (3, 10), rng)
    assert 'LinearOperatorMatrix(shape=(5, 3)' in repr(matrix)


def test_linearoperatormatrix_getitem():
    """Test slicing of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((12, 6), (3, 10), rng)

    def check(actual, expected):
        assert tuple(tuple(row) for row in actual) == tuple(tuple(row) for row in expected)

    sliced = matrix[1:3, 2]
    assert sliced.shape == (2, 1)
    check(sliced._operators, [row[2:3] for row in matrix._operators[1:3]])

    sliced = matrix[0]
    assert sliced.shape == (1, 6)
    check(sliced._operators, matrix._operators[:1])

    sliced = matrix[..., 0]
    assert sliced.shape == (12, 1)
    check(sliced._operators, [row[:1] for row in matrix._operators])

    sliced = matrix[1:6:2, (3, 4)]
    assert sliced.shape == (3, 2)
    check(sliced._operators, [[matrix._operators[i][j] for j in (3, 4)] for i in range(1, 6, 2)])


def test_linearoperatormarix_norm_rows():
    """Test norm of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((3, 1), (3, 10), rng)
    vector = rng.complex64_tensor((1, 10))
    result = matrix.operator_norm(*vector)
    expected = sum(row[0].operator_norm(vector[0], dim=None) ** 2 for row in matrix._operators) ** 0.5
    torch.testing.assert_close(result, expected)


def test_linearoperatormarix_norm_cols():
    """Test norm of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((1, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    result = matrix.operator_norm(*vector)
    expected = sum(op.operator_norm(v, dim=None) for op, v in zip(matrix._operators[0], vector, strict=False))
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_linearoperatormarix_norm(seed):
    """Test norm of LinearOperatorMatrix."""
    rng = RandomGenerator(seed)
    matrix = random_linearoperatormatrix((4, 2), (3, 10), rng)
    vector = rng.complex64_tensor((2, 10))
    result = matrix.operator_norm(*vector)

    class Wrapper(LinearOperator):
        """Stack the output of the matrix operator."""

        def forward(self, x):
            return (torch.stack(matrix(*x), 0),)

        def adjoint(self, x):
            return (torch.stack(matrix.adjoint(*x), 0),)

    real = Wrapper().operator_norm(vector, dim=None)

    assert result >= real
