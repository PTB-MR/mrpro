"""Test the analysis dictionary operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators import AnalysisDictionaryOp
from mrpro.utils import RandomGenerator

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)

SHAPE_PARAMETERS = pytest.mark.parametrize(
    ('dim', 'patches_shape', 'coefficient_shape'),
    [
        ((-2, -1), (3, 5, 1, 3, 2), (3, 5, 1, 256)),
        ((-3, -2, -1), (3, 5, 4, 3, 2), (3, 5, 256)),
        ((-1, -3, -2), (3, 5, 4, 3, 2), (3, 5, 256)),
        ((0, -2, -1), (4, 5, 1, 3, 2), (5, 1, 256)),
    ],
    ids=['2d patches', '3d patches', '3d patches random order', '2d+other patches'],
)


@SHAPE_PARAMETERS
def test_analysis_dictionary_op_adjointness(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test adjointness and shape of AnalysisDictionaryOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = AnalysisDictionaryOp(dictionary=dictionary, dim=dim)
    dotproduct_adjointness_test(operator, u, v)


@SHAPE_PARAMETERS
def test_analysis_dictionary_op_grad(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test gradient of AnalysisDictionaryOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = AnalysisDictionaryOp(dictionary=dictionary, dim=dim)
    gradient_of_linear_operator_test(operator, u, v)


@SHAPE_PARAMETERS
def test_analysis_dictionary_op_forward_mode_autodiff(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test forward-mode autodiff of AnalysisDictionaryOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = AnalysisDictionaryOp(dictionary=dictionary, dim=dim)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


def test_correlation_matrix_analysis_dicitionary_op() -> None:
    """Test that for a unit-norm synthesis dictionary the composition the forward to the complex-conjugated dictionary
    patches returns a correlation matrix with the diagonal entries corresponding to one."""
    rng = RandomGenerator(seed=0)
    n_atoms = 64
    dictionary_matrix = rng.randn_tensor(size=(n_atoms, 4, 6), dtype=torch.complex64)
    dictionary_matrix /= dictionary_matrix.norm(p=2, dim=(-2, -1), keepdim=True)
    analysis_op = AnalysisDictionaryOp(dictionary_matrix, dim=(-2, -1))
    (correlation_matrix,) = analysis_op(dictionary_matrix.conj())
    ones_tensor = torch.ones(n_atoms)

    torch.testing.assert_close(correlation_matrix.diag().abs(), ones_tensor)


def test_analysis_dictionary_op_raises_on_wrong_number_of_dims() -> None:
    """`dim` needs exactly one entry per patch axis of the dictionary."""
    dictionary = torch.randn(10, 5, 4, dtype=torch.complex64)

    with pytest.raises(ValueError, match='one entry per patch axis'):
        AnalysisDictionaryOp(dictionary, dim=-1)  # too few
    with pytest.raises(ValueError, match='one entry per patch axis'):
        AnalysisDictionaryOp(dictionary, dim=(-3, -2, -1))  # too many


def test_analysis_dictionary_op_raises_on_duplicate_dims() -> None:
    """`dim` must not refer to the same axis twice (checked on the literal indices)."""
    dictionary = torch.randn(10, 5, 4, dtype=torch.complex64)

    with pytest.raises(ValueError, match='duplicates'):
        AnalysisDictionaryOp(dictionary, dim=(-1, -1))


@pytest.mark.cuda
def test_analysis_dictionary_op_cuda() -> None:
    """Test DictionaryAnalysisOp works with CUDA devices."""
    # Generate random tensor
    dim = (0, -2, -1)
    patches_shape = (4, 5, 1, 3, 2)
    generator = RandomGenerator(seed=0)
    generate_tensor = generator.complex64_tensor
    u = generate_tensor(size=patches_shape)
    dictionary = generator.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))

    # Create on CPU, run on CPU
    dictionary_op = AnalysisDictionaryOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    (result,) = operator(u)
    assert result.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    dictionary_op = AnalysisDictionaryOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    operator.cuda()
    (result,) = operator(u.cuda())
    assert result.is_cuda
