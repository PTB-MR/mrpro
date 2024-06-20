"""Tests for linear operator."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import torch
from mrpro.operators import FastFourierOp

from tests import RandomGenerator


def test_autograd_wrapper():
    """Test the autograd wrapper using the fast fourier op as an example."""

    # Create test data, gradcheck requires double precision
    encoding_matrix = [5, 10, 15]
    recon_matrix = [3, 6, 9]
    generator = RandomGenerator(seed=0)
    u = generator.complex128_tensor(recon_matrix)
    v = generator.complex128_tensor(encoding_matrix)

    # Create operator
    ff_op = FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix)

    torch.autograd.gradcheck(ff_op.forward, u.requires_grad_())
    torch.autograd.gradcheck(ff_op.adjoint, v.requires_grad_())
