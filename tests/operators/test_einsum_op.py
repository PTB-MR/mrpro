"""Tests for Einsum Operator."""

from collections.abc import Sequence

import pytest
import torch
from mr2.operators.EinsumOp import EinsumOp
from mr2.utils import RandomGenerator

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_einsum_op_and_range_domain(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: torch.dtype
) -> tuple[EinsumOp, torch.Tensor, torch.Tensor]:
    """Create an Einsum operator and an element from range and domain."""
    rng = RandomGenerator(seed=0)
    tensor = rng.rand_tensor(size=tensor_shape, dtype=dtype)
    u = rng.rand_tensor(size=input_shape, dtype=dtype)
    v = rng.rand_tensor(size=output_shape, dtype=dtype)
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


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex128], ids=['float32', 'complex128'])
@EINSUM_PARAMETERS
def test_einsum_op(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: torch.dtype
) -> None:
    """Test adjointness and shape."""
    dotproduct_adjointness_test(
        *create_einsum_op_and_range_domain(tensor_shape, input_shape, rule, output_shape, dtype)
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex128], ids=['float32', 'complex128'])
@EINSUM_PARAMETERS
def test_einsum_op_grad(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: torch.dtype
) -> None:
    """Test the gradient of the einsum operator."""
    gradient_of_linear_operator_test(
        *create_einsum_op_and_range_domain(tensor_shape, input_shape, rule, output_shape, dtype)
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex128], ids=['float32', 'complex128'])
@EINSUM_PARAMETERS
def test_einsum_op_forward_mode_autodiff(
    tensor_shape: Sequence[int], input_shape: Sequence[int], rule: str, output_shape: Sequence[int], dtype: torch.dtype
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


@pytest.mark.cuda
def test_einsum_op_cuda() -> None:
    """Test einsum operator works on cuda devices."""
    tensor_shape = (3, 5, 4, 2)
    input_shape = (3, 2, 5)
    rule = 'l ... i j, l j k -> k i l'
    generator = RandomGenerator(seed=0)
    generate_tensor = generator.complex128_tensor
    tensor = generate_tensor(size=tensor_shape)
    input_tensor = generate_tensor(size=input_shape)

    # Create on CPU, transfer to GPU, run on GPU
    einsum_op = EinsumOp(tensor, rule)
    operator = einsum_op.H @ einsum_op
    operator.cuda()
    (output_tensor,) = operator(input_tensor.cuda())
    assert output_tensor.is_cuda

    # Create on CPU, run on CPU
    einsum_op = EinsumOp(tensor, rule)
    operator = einsum_op.H @ einsum_op
    (output_tensor,) = operator(input_tensor)
    assert output_tensor.is_cpu

    # Create on GPU, run on GPU
    einsum_op = EinsumOp(tensor.cuda(), rule)
    operator = einsum_op.H @ einsum_op
    (output_tensor,) = operator(input_tensor.cuda())
    assert output_tensor.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    einsum_op = EinsumOp(tensor.cuda(), rule)
    operator = einsum_op.H @ einsum_op
    operator.cpu()
    (output_tensor,) = operator(input_tensor)
    assert output_tensor.is_cpu
