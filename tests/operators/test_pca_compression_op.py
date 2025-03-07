"""Tests for PCA Compression Operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators import PCACompressionOp

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_pca_compression_op_and_domain_range(
    init_data_shape: Sequence[int], input_shape: Sequence[int], n_components: int
) -> tuple[PCACompressionOp, torch.Tensor, torch.Tensor]:
    """Create a pca compression operator and an element from domain and range."""
    # Create test data
    generator = RandomGenerator(seed=0)
    data_to_calculate_compression_matrix_from = generator.complex64_tensor(init_data_shape)
    u = generator.complex64_tensor(input_shape)
    output_shape = (*input_shape[:-1], n_components)
    v = generator.complex64_tensor(output_shape)

    # Create operator and apply
    pca_comp_op = PCACompressionOp(data=data_to_calculate_compression_matrix_from, n_components=n_components)
    return pca_comp_op, u, v


SHAPE_PARAMETERS = pytest.mark.parametrize(
    ('init_data_shape', 'input_shape', 'n_components'),
    [
        ((40, 10), (100, 10), 6),
        ((40, 10), (3, 4, 5, 100, 10), 3),
        ((3, 4, 40, 10), (3, 4, 100, 10), 6),
        ((3, 4, 40, 10), (7, 3, 4, 100, 10), 3),
    ],
)


@SHAPE_PARAMETERS
def test_pca_compression_op_adjoint(
    init_data_shape: Sequence[int], input_shape: Sequence[int], n_components: int
) -> None:
    """Test adjointness of PCA Compression Op."""
    dotproduct_adjointness_test(*create_pca_compression_op_and_domain_range(init_data_shape, input_shape, n_components))


@SHAPE_PARAMETERS
def test_pca_compression_op_grad(init_data_shape: Sequence[int], input_shape: Sequence[int], n_components: int) -> None:
    """Test gradient of PCA Compression Op."""
    gradient_of_linear_operator_test(
        *create_pca_compression_op_and_domain_range(init_data_shape, input_shape, n_components)
    )


@SHAPE_PARAMETERS
def test_pca_compression_op_forward_mode_autodiff(
    init_data_shape: Sequence[int], input_shape: Sequence[int], n_components: int
) -> None:
    """Test forward-mode autodiff of PCA Compression Op."""
    forward_mode_autodiff_of_linear_operator_test(
        *create_pca_compression_op_and_domain_range(init_data_shape, input_shape, n_components)
    )


def test_pca_compression_op_wrong_shapes() -> None:
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


@pytest.mark.cuda
def test_pca_compression_op_cuda() -> None:
    """Test if PCA compression operator works on CUDA devices."""
    init_data_shape = (40, 10)
    n_components = 6

    # Create test data
    generator = RandomGenerator(seed=0)
    data_to_calculate_compression_matrix_from = generator.complex64_tensor(init_data_shape)

    # Create on CPU, transfer to GPU, run on GPU
    pca_comp_op = PCACompressionOp(data=data_to_calculate_compression_matrix_from, n_components=n_components)
    pca_comp_op.cuda()
    (comp_result,) = pca_comp_op(data_to_calculate_compression_matrix_from.cuda())
    assert comp_result.is_cuda

    # Create on CPU, run on CPU
    pca_comp_op = PCACompressionOp(data=data_to_calculate_compression_matrix_from, n_components=n_components)
    (comp_result,) = pca_comp_op(data_to_calculate_compression_matrix_from)
    assert comp_result.is_cpu

    # Create on GPU, run on GPU
    pca_comp_op = PCACompressionOp(data=data_to_calculate_compression_matrix_from.cuda(), n_components=n_components)
    (comp_result,) = pca_comp_op(data_to_calculate_compression_matrix_from.cuda())
    assert comp_result.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    pca_comp_op = PCACompressionOp(data=data_to_calculate_compression_matrix_from.cuda(), n_components=n_components)
    pca_comp_op.cpu()
    (comp_result,) = pca_comp_op(data_to_calculate_compression_matrix_from)
    assert comp_result.is_cpu
