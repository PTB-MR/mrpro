"""Tests for Einsum Operator."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pytest
import torch
from mrpro.operators._EinsumOp import EinsumOp

from tests import RandomGenerator
from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize('dtype', ['float32', 'complex128'])
@pytest.mark.parametrize(
    ('tensor_shape', 'input_shape', 'rule', 'output_shape'),
    [
        ((3, 3), (3), 'ij,j->i', (3)),  # matrix vector product
        ((2, 4), (2, 4), '...i,...i->...i', (2, 4)),  # hadamard product
        ((2, 4, 3), (2, 3), '...ij, ...j->...i', (2, 4)),  # batched matrix product
        ((4, 3), (3, 2), '... i j , j k -> i k', (4, 2)),  # additional spaces in rule
        ((3, 5, 4, 2), (3, 2, 5), 'l...ij, ljk -> kil', (5, 4, 3)),  # general tensor contraction
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
