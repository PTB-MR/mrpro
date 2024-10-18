"""Tests for PCA Compression Operator."""

import pytest
from mrpro.operators import PCACompressionOp

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize(
    ('init_data_shape', 'input_shape', 'n_components', 'compression_dim', 'separate_dims'),
    [
        ((10, 100), (10, 15, 30, 25), 3, 0, None),
        ((10, 15, 100), (10, 15, 30, 25), 6, 0, (1,)),
        ((3, 10, 20, 25), (4, 3, 10, 30, 25), 2, -3, (-4,)),
        ((3, 10, 20, 25), (4, 3, 10, 30, 25), 4, -1, (-3,)),
        ((3, 10, 20, 25), (4, 3, 10, 20, 25), 7, -3, (-2, -1)),
    ],
)
def test_pca_compression_op_adjoint(init_data_shape, input_shape, n_components, compression_dim, separate_dims):
    """Test adjointness of PCA Compression Op."""

    # Create test data
    generator = RandomGenerator(seed=0)
    data_to_calculate_compression_matrix = generator.complex64_tensor(init_data_shape)
    u = generator.complex64_tensor(input_shape)
    output_shape = list(input_shape)
    output_shape[compression_dim] = n_components
    v = generator.complex64_tensor(output_shape)

    # Create operator and apply
    pca_comp_op = PCACompressionOp(
        data=data_to_calculate_compression_matrix,
        n_components=n_components,
        compression_dim=compression_dim,
        separate_dims=separate_dims,
    )
    dotproduct_adjointness_test(pca_comp_op, u, v)


def test_pca_compression_op_wrong_compression_dim():
    """Raise error for compression dim in separate_dims."""
    input_shape = (5, 10, 20, 30)

    # Create test data
    generator = RandomGenerator(seed=0)
    input_data = generator.complex64_tensor(input_shape)

    with pytest.raises(ValueError, match='compression dimension must not be in separate_dim'):
        PCACompressionOp(data=input_data, n_components=2, compression_dim=-2, separate_dims=(-2, -3))
