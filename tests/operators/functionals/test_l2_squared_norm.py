"""Tests for L2-Squared-functional."""

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
from mrpro.operators.functionals.l2_squared import L2NormSquared

@pytest.mark.parametrize(
    ('x', 'expected_result_x','expected_result_p','expected_result_pcc', 'expected_result_p_forward', 'expected_result_pcc_forward'),
    [
        (torch.tensor([1,1],dtype=torch.complex64), torch.tensor([2.0]), torch.tensor([1/3,1/3],dtype=torch.complex64),
        torch.tensor([2/3,2/3],dtype=torch.complex64), torch.tensor([1/3**2+1/3**2],dtype=torch.complex64), torch.tensor([2/3**2+2/3**2])),
        (torch.tensor([1+1j,1+1j],dtype=torch.complex64), torch.tensor([4.0]), torch.tensor([(1+1j)/3,(1+1j)/3],dtype=torch.complex64),
        torch.tensor([2*(1+1j)/3,2*(1+1j)/3],dtype=torch.complex64), torch.tensor([((1+1j)/3)**2+((1+1j)/3)**2],dtype=torch.complex64), torch.tensor([2/3**2+2/3**2])),
    ],
)
def test_l2_squared_functional(x, expected_result_x, expected_result_p, expected_result_pcc, expected_result_p_forward, expected_result_pcc_forward):
    """Test if mse_data_discrepancy matches expected values.

    Expected values are supposed to be
    1/N*|| . - data||_2^2
    """
    l2_squared = L2NormSquared(lam=1)
    # prox + forward
    (p,) = l2_squared.prox(x, sigma=1)
    (p_forward,) = l2_squared.forward(p)
    #forward
    (x_forward,) = l2_squared.forward(x)
    # prox convex conjugate
    (pcc,) = l2_squared.prox_convex_conj(x, sigma=1)
    (pcc_forward,) = l2_squared.forward(pcc)

    torch.testing.assert_close(x_forward, expected_result_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(p, expected_result_p, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(pcc, expected_result_pcc, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(p_forward, expected_result_p_forward, rtol=1e-3, atol=1e-3)
    #torch.testing.assert_close(pcc_forward, expected_result_pcc_forward, rtol=1e-3, atol=1e-3)
