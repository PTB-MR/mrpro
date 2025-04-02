"""Tests for Rearrange Operator."""

from collections.abc import Sequence

import pytest
from mrpro.operators.RearrangeOp import RearrangeOp

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)

SHAPE_PARAMETERS = pytest.mark.parametrize(
    ('input_shape', 'rule', 'output_shape', 'additional_info'),
    [
        ((1, 2, 3), 'a b c-> b a c', (2, 1, 3), None),  # swap axes
        ((2, 2, 4), '... a b->... (a b)', (2, 8), {'b': 4}),  # flatten
        ((2), '... (a b) -> ... a b', (2, 1), {'b': 1}),  # unflatten
    ],
    ids=['swap_axes', 'flatten', 'unflatten'],
)


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@SHAPE_PARAMETERS
def test_rearrange_op_adjointness(
    input_shape: Sequence[int], rule: str, output_shape: Sequence[int], additional_info: dict, dtype: str
) -> None:
    """Test adjointness and shape of Rearrange Op."""
    generator = RandomGenerator(seed=0)
    generate_tensor = getattr(generator, f'{dtype}_tensor')
    u = generate_tensor(size=input_shape)
    v = generate_tensor(size=output_shape)
    operator = RearrangeOp(rule, additional_info)
    dotproduct_adjointness_test(operator, u, v)


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@SHAPE_PARAMETERS
def test_rearrange_op_grad(
    input_shape: Sequence[int], rule: str, output_shape: Sequence[int], additional_info: dict, dtype: str
) -> None:
    """Test gradient of Rearrange Op."""
    generator = RandomGenerator(seed=0)
    generate_tensor = getattr(generator, f'{dtype}_tensor')
    u = generate_tensor(size=input_shape)
    v = generate_tensor(size=output_shape)
    operator = RearrangeOp(rule, additional_info)
    gradient_of_linear_operator_test(operator, u, v)


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@SHAPE_PARAMETERS
def test_rearrange_op_forward_mode_autodiff(
    input_shape: Sequence[int], rule: str, output_shape: Sequence[int], additional_info: dict, dtype: str
) -> None:
    """Test forward-mode autodiff of Rearrange Op."""
    generator = RandomGenerator(seed=0)
    generate_tensor = getattr(generator, f'{dtype}_tensor')
    u = generate_tensor(size=input_shape)
    v = generate_tensor(size=output_shape)
    operator = RearrangeOp(rule, additional_info)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


def test_rearrange_op_invalid() -> None:
    """Test with invalid rule."""
    with pytest.raises(ValueError, match='pattern should match'):
        RearrangeOp('missing arrow')
