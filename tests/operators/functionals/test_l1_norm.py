"""Tests for L1-functional."""

# Copyright 20234 Physikalisch-Technische Bundesanstalt
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

import pytest
import torch
from mrpro.operators.functionals.l1 import L1Norm


@pytest.mark.parametrize(
    (
        'x',
        'expected_result_x',
        'expected_result_p',
        'expected_result_pcc',
        'expected_result_p_forward',
        'expected_result_pcc_forward',
    ),
    [
        (
            torch.tensor([1, 1], dtype=torch.complex64),
            torch.tensor([2.0]),
            torch.tensor([1 + 0j, 1 + 0j], dtype=torch.complex64),
            torch.tensor([1 + 0j, 1 + 0j], dtype=torch.complex64),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([2.0]),
        ),
        (
            torch.tensor([1 + 1j, 1 + 1j], dtype=torch.complex64),
            torch.tensor([2.8284]),
            torch.tensor([1.0 + 1.0j, 1.0 + 1.0j], dtype=torch.complex64),
            torch.tensor([1.0 + 1.0j, 1.0 + 1.0j], dtype=torch.complex64),
            torch.tensor([2.8284], dtype=torch.float32),
            torch.tensor([2.8284]),
        ),
        (
            torch.tensor([1 + 0j, 1 + 1j], dtype=torch.complex64),
            torch.tensor([2.4142], dtype=torch.float32),
            torch.tensor([1.0 + 0.0j, 1.0 + 1.0j], dtype=torch.complex64),
            torch.tensor([1.0 + 0.0j, 1.0 + 1.0j], dtype=torch.complex64),
            torch.tensor([2.4142], dtype=torch.float32),
            torch.tensor([2.4142]),
        ),
    ],
)
def test_l1_functional(
    x, expected_result_x, expected_result_p, expected_result_pcc, expected_result_p_forward, expected_result_pcc_forward
):
    """Test if L1 norm matches expected values."""
    l1_norm = L1Norm(lam=1)
    # prox + forward
    (p,) = l1_norm.prox(x, sigma=1)
    (p_forward,) = l1_norm.forward(p)
    # forward
    (x_forward,) = l1_norm.forward(x)
    # prox convex conjugate
    (pcc,) = l1_norm.prox_convex_conj(x, sigma=1)
    (pcc_forward,) = l1_norm.forward(pcc)

    torch.testing.assert_close(x_forward, expected_result_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(p, expected_result_p, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(pcc, expected_result_pcc, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(p_forward, expected_result_p_forward, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(pcc_forward, expected_result_pcc_forward, rtol=1e-3, atol=1e-3)
