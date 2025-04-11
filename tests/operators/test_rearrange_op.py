"""Tests for Rearrange Operator."""

from collections.abc import Sequence

import pytest
import torch
from mrpro.operators.RearrangeOp import RearrangeOp
from mrpro.utils import RandomGenerator

from tests import (
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


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex128], ids=['float32', 'complex128'])
@SHAPE_PARAMETERS
def test_rearrange_op_adjointness(
    input_shape: Sequence[int], rule: str, output_shape: Sequence[int], additional_info: dict, dtype: torch.dtype
) -> None:
    """Test adjointness and shape of Rearrange Op."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=input_shape, dtype=dtype)
    v = rng.rand_tensor(size=output_shape, dtype=dtype)
    operator = RearrangeOp(rule, additional_info)
    dotproduct_adjointness_test(operator, u, v)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex128], ids=['float32', 'complex128'])
@SHAPE_PARAMETERS
def test_rearrange_op_grad(
    input_shape: Sequence[int], rule: str, output_shape: Sequence[int], additional_info: dict, dtype: torch.dtype
) -> None:
    """Test gradient of Rearrange Op."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=input_shape, dtype=dtype)
    v = rng.rand_tensor(size=output_shape, dtype=dtype)
    operator = RearrangeOp(rule, additional_info)
    gradient_of_linear_operator_test(operator, u, v)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex128], ids=['float32', 'complex128'])
@SHAPE_PARAMETERS
def test_rearrange_op_forward_mode_autodiff(
    input_shape: Sequence[int], rule: str, output_shape: Sequence[int], additional_info: dict, dtype: torch.dtype
) -> None:
    """Test forward-mode autodiff of Rearrange Op."""
    rng = RandomGenerator(seed=0)
    u = rng.rand_tensor(size=input_shape, dtype=dtype)
    v = rng.rand_tensor(size=output_shape, dtype=dtype)
    operator = RearrangeOp(rule, additional_info)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


def test_rearrange_op_invalid() -> None:
    """Test with invalid rule."""
    with pytest.raises(ValueError, match='pattern should match'):
        RearrangeOp('missing arrow')
