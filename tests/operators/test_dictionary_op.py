"""Test the dictionary operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators import DictionaryOp
from mrpro.utils import RandomGenerator

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)

SHAPE_PARAMETERS = pytest.mark.parametrize(
    ('dim', 'input_shape', 'output_shape'),
    [
        ((-2, -1), (3, 5, 1, 3, 2), (3, 5, 1, 256)),
        ((-3, -2, -1), (3, 5, 4, 3, 2), (3, 5, 256)),
        ((0, -2, -1), (4, 5, 1, 3, 2), (5, 1, 256)),
    ],
    ids=['2d patches', '3d patches', '2d+other patches'],
)


@SHAPE_PARAMETERS
def test_dictionary_op_adjointness(dim: Sequence[int], input_shape: Sequence[int], output_shape: Sequence[int]) -> None:
    """Test adjointness and shape of Dictionary Op."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=input_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=output_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[input_shape[d] for d in dim]))
    operator = DictionaryOp(dictionary=dictionary, dim=dim)
    dotproduct_adjointness_test(operator, u, v)


@SHAPE_PARAMETERS
def test_dictionary_op_grad(dim: Sequence[int], input_shape: Sequence[int], output_shape: Sequence[int]) -> None:
    """Test gradient of Dictionary Op."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=input_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=output_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[input_shape[d] for d in dim]))
    operator = DictionaryOp(dictionary=dictionary, dim=dim)
    gradient_of_linear_operator_test(operator, u, v)


@SHAPE_PARAMETERS
def test_dictionary_op_forward_mode_autodiff(
    dim: Sequence[int], input_shape: Sequence[int], output_shape: Sequence[int]
) -> None:
    """Test forward-mode autodiff of Dictionary Op."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=input_shape, dtype=torch.complex64)
    v = rng.rand_tensor(size=output_shape, dtype=torch.complex64)
    dictionary = rng.complex64_tensor(size=(256, *[input_shape[d] for d in dim]))
    operator = DictionaryOp(dictionary=dictionary, dim=dim)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


@pytest.mark.cuda
def test_dictionary_op_cuda() -> None:
    """Test Dictionary operator works with CUDA devices."""
    # Generate random tensor
    dim = (0, -2, -1)
    input_shape = (4, 5, 1, 3, 2)
    generator = RandomGenerator(seed=0)
    generate_tensor = generator.complex64_tensor
    u = generate_tensor(size=input_shape)
    dictionary = generator.complex64_tensor(size=(256, *[input_shape[d] for d in dim]))

    # Create on CPU, run on CPU
    dictionary_op = DictionaryOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    (result,) = operator(u)
    assert result.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    dictionary_op = DictionaryOp(dictionary=dictionary, dim=dim)
    operator = dictionary_op.H @ dictionary_op
    operator.cuda()
    (result,) = operator(u.cuda())
    assert result.is_cuda
