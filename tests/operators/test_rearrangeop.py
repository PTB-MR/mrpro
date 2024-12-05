"""Tests for Rearrange Operator."""

import pytest
from mrpro.operators.RearrangeOp import RearrangeOp

from tests import RandomGenerator, dotproduct_adjointness_test


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@pytest.mark.parametrize(
    ('input_shape', 'rule', 'output_shape', 'additional_info'),
    [
        ((1, 2, 3), 'a b c-> b a c', (2, 1, 3), None),  # swap axes
        ((2, 2, 4), '... a b->... (a b)', (2, 8), {'b': 4}),  # flatten
        ((2), '... (a b) -> ... a b', (2, 1), {'b': 1}),  # unflatten
    ],
    ids=['swap_axes', 'flatten', 'unflatten'],
)
def test_einsum_op(input_shape, rule, output_shape, additional_info, dtype):
    """Test adjointness and shape."""
    generator = RandomGenerator(seed=0)
    generate_tensor = getattr(generator, f'{dtype}_tensor')
    u = generate_tensor(size=input_shape)
    v = generate_tensor(size=output_shape)
    operator = RearrangeOp(rule, additional_info)
    dotproduct_adjointness_test(operator, u, v)


def test_einsum_op_invalid():
    """Test with invalid rule."""
    with pytest.raises(ValueError, match='pattern should match'):
        RearrangeOp('missing arrow')
