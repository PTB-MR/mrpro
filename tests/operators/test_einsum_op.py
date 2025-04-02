"""Tests for Einsum Operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.EinsumOp import EinsumOp

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_einsum_op_and_range_domain(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: str
) -> tuple[EinsumOp, torch.Tensor, torch.Tensor]:
    """Create an Einsum operator and an element from range and domain."""
    generator = RandomGenerator(seed=0)
    generate_tensor = getattr(generator, f'{dtype}_tensor')
    tensor = generate_tensor(size=tensor_shape)
    u = generate_tensor(size=input_shape)
    v = generate_tensor(size=output_shape)
    einsum_op = EinsumOp(tensor, rule)
    return einsum_op, u, v


EINSUM_PARAMETERS = pytest.mark.parametrize(
    ('tensor_shape', 'input_shape', 'rule', 'output_shape'),
    [
        ((3, 3), (3), 'i j,j->i', (3)),  # matrix vector product
        ((2, 4), (2, 4), '... i, ... i->... i', (2, 4)),  # hadamard product
        ((2, 4, 3), (2, 3), '... i j, ... j->... i', (2, 4)),  # batched matrix product
        ((4, 3), (3, 2), '... i j , j k -> i k', (4, 2)),  # additional spaces in rule
        ((3, 5, 4, 2), (3, 2, 5), 'l ... i j, l j k -> k i l', (5, 4, 3)),  # general tensor contraction
    ],
)


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@EINSUM_PARAMETERS
def test_einsum_op(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: str
) -> None:
    """Test adjointness and shape."""
    dotproduct_adjointness_test(
        *create_einsum_op_and_range_domain(tensor_shape, input_shape, rule, output_shape, dtype)
    )


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@EINSUM_PARAMETERS
def test_einsum_op_grad(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: str
) -> None:
    """Test the gradient of the einsum operator."""
    gradient_of_linear_operator_test(
        *create_einsum_op_and_range_domain(tensor_shape, input_shape, rule, output_shape, dtype)
    )


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@EINSUM_PARAMETERS
def test_einsum_op_forward_mode_autodiff(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: str
) -> None:
    """Test forward-mode autodiff of the einsum operator."""
    forward_mode_autodiff_of_linear_operator_test(
        *create_einsum_op_and_range_domain(tensor_shape, input_shape, rule, output_shape, dtype)
    )


@pytest.mark.parametrize(
    'rule',
    [
        'no -> coma',  # missing comma
        'no, arow',  # missing arrow
        ',no->first',  # missing first argument
        'no->second',  # missing second argument
        '',  # empty string
    ],
)
def test_einsum_op_invalid(rule: str) -> None:
    """Test with different invalid rules."""
    with pytest.raises(ValueError, match='pattern should match'):
        EinsumOp(torch.tensor([]), rule)
