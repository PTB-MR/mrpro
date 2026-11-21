"""Test the synthesis dictionary operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators import AnalysisDictionaryOp, SynthesisDictionaryOp
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
def test_synthesis_dictionary_op_adjointness(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test adjointness and shape of SynthesisDictionaryOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = SynthesisDictionaryOp(dictionary=dictionary, dim=dim)
    dotproduct_adjointness_test(operator, u, v)


@SHAPE_PARAMETERS
def test_synthesis_dictionary_op_grad(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test gradient of SynthesisDictionaryOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = SynthesisDictionaryOp(dictionary=dictionary, dim=dim)
    gradient_of_linear_operator_test(operator, u, v)


@SHAPE_PARAMETERS
def test_synthesis_dictionary_op_forward_mode_autodiff(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test forward-mode autodiff of SynthesisDictionaryOp."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))
    operator = SynthesisDictionaryOp(dictionary=dictionary, dim=dim)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


@SHAPE_PARAMETERS
def test_analysis_dictionary_adjoint_of_synthesis_dictionary(
    dim: Sequence[int], patches_shape: Sequence[int], coefficient_shape: Sequence[int]
) -> None:
    """Test that the synthesis and analysis operators are adjoints of each other."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=coefficient_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=patches_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))

    synthesis_op = SynthesisDictionaryOp(dictionary=dictionary.conj(), dim=dim)
    analysis_op = AnalysisDictionaryOp(dictionary=dictionary, dim=dim)

    torch.testing.assert_close(synthesis_op(u)[0], analysis_op.H(u)[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(synthesis_op.H(v)[0], analysis_op(v)[0], rtol=1e-2, atol=1e-2)


@pytest.mark.cuda
def test_synthesis_dictionary_op_cuda() -> None:
    """Test SynthesisDictionaryOp works with CUDA devices."""
    # Generate random tensor
    dim = (0, -2, -1)
    patches_shape = (4, 5, 1, 3, 2)
    coefficient_shape = (5, 1, 256)
    generator = RandomGenerator(seed=0)
    generate_tensor = generator.complex64_tensor
    u = generate_tensor(size=coefficient_shape)
    dictionary = generator.complex64_tensor(size=(256, *[patches_shape[d] for d in dim]))

    # Create on CPU, run on CPU
    dictionary_op = SynthesisDictionaryOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    (result,) = operator(u)
    assert result.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    dictionary_op = SynthesisDictionaryOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    operator.cuda()
    (result,) = operator(u.cuda())
    assert result.is_cuda
