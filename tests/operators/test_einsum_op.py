"""Tests for Einsum Operator."""

import pytest
import torch
from mrpro.operators.EinsumOp import EinsumOp

from tests import RandomGenerator, dotproduct_adjointness_test


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@pytest.mark.parametrize(
    ('tensor_shape', 'input_shape', 'rule', 'output_shape'),
    [
        ((3, 3), (3), 'i j,j->i', (3)),  # matrix vector product
        ((2, 4), (2, 4), '... i, ... i->... i', (2, 4)),  # hadamard product
        ((2, 4, 3), (2, 3), '... i j, ... j->... i', (2, 4)),  # batched matrix product
        ((4, 3), (3, 2), '... i j , j k -> i k', (4, 2)),  # additional spaces in rule
        ((3, 5, 4, 2), (3, 2, 5), 'l ... i j, l j k -> k i l', (5, 4, 3)),  # general tensor contraction
    ],
)
def test_einsum_op(tensor_shape, input_shape, rule, output_shape, dtype):
    """Test adjointness and shape."""
    generator = RandomGenerator(seed=0)
    generate_tensor = getattr(generator, f'{dtype}_tensor')
    tensor = generate_tensor(size=tensor_shape)
    u = generate_tensor(size=input_shape)
    v = generate_tensor(size=output_shape)
    operator = EinsumOp(tensor, rule)
    dotproduct_adjointness_test(operator, u, v)


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
def test_einsum_op_invalid(rule):
    """Test with different invalid rules."""
    with pytest.raises(ValueError, match='pattern should match'):
        EinsumOp(torch.tensor([]), rule)


@pytest.mark.cuda
def test_einsum_op_cuda():
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
    einsum_op.cuda()
    (output_tensor,) = einsum_op(input_tensor.cuda())
    assert output_tensor.is_cuda

    # Create on CPU, run on CPU
    einsum_op = EinsumOp(tensor, rule)
    (output_tensor,) = einsum_op(input_tensor)
    assert output_tensor.is_cpu

    # Create on GPU, run on GPU
    einsum_op = EinsumOp(tensor.cuda(), rule)
    (output_tensor,) = einsum_op(input_tensor.cuda())
    assert output_tensor.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    einsum_op = EinsumOp(tensor.cuda(), rule)
    einsum_op.cpu()
    (output_tensor,) = einsum_op(input_tensor)
    assert output_tensor.is_cpu
