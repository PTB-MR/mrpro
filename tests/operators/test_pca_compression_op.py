"""Tests for PCA Compression Operator."""

import pytest
from mrpro.operators import PCACompressionOp

from tests import RandomGenerator, dotproduct_adjointness_test


@pytest.mark.parametrize(
    ('init_data_shape', 'input_shape', 'n_components'),
    [
        ((40, 10), (100, 10), 6),
        ((40, 10), (3, 4, 5, 100, 10), 3),
        ((3, 4, 40, 10), (3, 4, 100, 10), 6),
        ((3, 4, 40, 10), (7, 3, 4, 100, 10), 3),
    ],
)
def test_pca_compression_op_adjoint(init_data_shape, input_shape, n_components):
    """Test adjointness of PCA Compression Op."""

    # Create test data
    generator = RandomGenerator(seed=0)
    data_to_calculate_compression_matrix_from = generator.complex64_tensor(init_data_shape)
    u = generator.complex64_tensor(input_shape)
    output_shape = (*input_shape[:-1], n_components)
    v = generator.complex64_tensor(output_shape)

    # Create operator and apply
    pca_comp_op = PCACompressionOp(data=data_to_calculate_compression_matrix_from, n_components=n_components)
    dotproduct_adjointness_test(pca_comp_op, u, v)


def test_pca_compression_op_wrong_shapes():
    """Test if Operator raises error if shape mismatch."""
    init_data_shape = (10, 6)
    input_shape = (100, 3)

    # Create test data
    generator = RandomGenerator(seed=0)
    data_to_calculate_compression_matrix_from = generator.complex64_tensor(init_data_shape)
    input_data = generator.complex64_tensor(input_shape)

    pca_comp_op = PCACompressionOp(data=data_to_calculate_compression_matrix_from, n_components=2)

    with pytest.raises(RuntimeError, match='Matrix'):
        pca_comp_op(input_data)

    with pytest.raises(RuntimeError, match='Matrix.H'):
        pca_comp_op.adjoint(input_data)
