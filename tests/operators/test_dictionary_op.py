"""Test the dictionary analysis and synthesis operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators import DictionaryAnalysisOp, DictionarySynthesisOp
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
def test_dictionary_analysis_op_adjointness(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test adjointness and shape of DictionaryAnalysisOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = DictionaryAnalysisOp(dictionary=dictionary, dim=dim)
    dotproduct_adjointness_test(operator, u, v)


@SHAPE_PARAMETERS
def test_dictionary_analysis_op_grad(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test gradient of DictionaryAnalysisOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = DictionaryAnalysisOp(dictionary=dictionary, dim=dim)
    gradient_of_linear_operator_test(operator, u, v)


@SHAPE_PARAMETERS
def test_dictionary_analysis_op_forward_mode_autodiff(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test forward-mode autodiff of DictionaryAnalysisOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = DictionaryAnalysisOp(dictionary=dictionary, dim=dim)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


@pytest.mark.parametrize('k', [0, 3, 9])
def test_dictionary_analysis_op_conjugates_dictionary(k: int) -> None:
    r"""Analysis of a dictionary entry returns its squared norm at its own index.

    The analysis computes :math:`c_i = \langle d_i, x \rangle = \sum_j \overline{d_i[j]}\, x[j]`.
    For :math:`x = d_k` this gives

    .. math::
        c_k = \sum_j \overline{d_k[j]}\, d_k[j] = \sum_j |d_k[j]|^2 = \|d_k\|_2^2,

    which is real and non-negative. Without the conjugate, the result would be :math:`\sum_j d_k[j]^2`,
    which is generally complex. By Cauchy-Schwarz, :math:`|c_i| \le \|d_i\|\,\|d_k\|` for all other entries.
    """
    torch.manual_seed(0)
    n_entries = 10
    dictionary = torch.randn(n_entries, 5, 4, dtype=torch.complex64)
    analysis_op = DictionaryAnalysisOp(dictionary, dim=(-2, -1))

    (coefficients,) = analysis_op(dictionary[k])

    # c_k = ||d_k||^2: real and non-negative
    squared_norm = dictionary[k].abs().square().sum()
    torch.testing.assert_close(coefficients[k], squared_norm.to(coefficients.dtype))
    assert coefficients[k].imag.abs() < 1e-4 * squared_norm

    # Cauchy-Schwarz: |c_i| <= ||d_i|| ||d_k|| for all entries, so no entry exceeds the self-match
    norms = dictionary.flatten(1).norm(dim=1)
    assert torch.all(coefficients.abs() <= norms * norms[k] * (1 + 1e-5))


def test_dictionary_op_raises_on_wrong_number_of_dims() -> None:
    """`dim` needs exactly one entry per patch axis of the dictionary."""
    dictionary = torch.randn(10, 5, 4, dtype=torch.complex64)

    with pytest.raises(ValueError, match='one entry per patch axis'):
        DictionaryAnalysisOp(dictionary, dim=-1)  # too few
    with pytest.raises(ValueError, match='one entry per patch axis'):
        DictionaryAnalysisOp(dictionary, dim=(-3, -2, -1))  # too many


def test_dictionary_op_raises_on_duplicate_dims() -> None:
    """`dim` must not refer to the same axis twice (checked on the literal indices)."""
    dictionary = torch.randn(10, 5, 4, dtype=torch.complex64)

    with pytest.raises(ValueError, match='duplicates'):
        DictionaryAnalysisOp(dictionary, dim=(-1, -1))


@pytest.mark.cuda
def test_dictionary_analysis_op_cuda() -> None:
    """Test DictionaryAnalysisOp works with CUDA devices."""
    # Generate random tensor
    dim = (0, -2, -1)
    patches_shape = (4, 5, 1, 3, 2)
    generator = RandomGenerator(seed=0)
    generate_tensor = generator.complex64_tensor
    u = generate_tensor(size=patches_shape)
    dictionary = generator.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))

    # Create on CPU, run on CPU
    dictionary_op = DictionaryAnalysisOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    (result,) = operator(u)
    assert result.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    dictionary_op = DictionaryAnalysisOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    operator.cuda()
    (result,) = operator(u.cuda())
    assert result.is_cuda


@SHAPE_PARAMETERS
def test_dictionary_synthesis_op_adjointness(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test adjointness and shape of DictionarySynthesisOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = DictionarySynthesisOp(dictionary=dictionary, dim=dim)
    dotproduct_adjointness_test(operator, u, v)


@SHAPE_PARAMETERS
def test_dictionary_synthesis_op_grad(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test gradient of DictionarySynthesisOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = DictionarySynthesisOp(dictionary=dictionary, dim=dim)
    gradient_of_linear_operator_test(operator, u, v)


@SHAPE_PARAMETERS
def test_dictionary_synthesis_op_forward_mode_autodiff(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test forward-mode autodiff of DictionarySynthesisOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = DictionarySynthesisOp(dictionary=dictionary, dim=dim)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


@pytest.mark.cuda
def test_dictionary_synthesis_op_cuda() -> None:
    """Test DictionarySynthesisOp works with CUDA devices."""
    # Generate random tensor
    dim = (0, -2, -1)
    patches_shape = (4, 5, 1, 3, 2)
    coefficient_shape = (5, 1, 256)
    generator = RandomGenerator(seed=0)
    generate_tensor = generator.complex64_tensor
    u = generate_tensor(size=coefficient_shape)
    dictionary = generator.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))

    # Create on CPU, run on CPU
    dictionary_op = DictionarySynthesisOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    (result,) = operator(u)
    assert result.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    dictionary_op = DictionarySynthesisOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    operator.cuda()
    (result,) = operator(u.cuda())
    assert result.is_cuda
