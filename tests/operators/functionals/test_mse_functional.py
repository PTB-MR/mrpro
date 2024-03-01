"""Tests for MSE-functional."""

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
from mrpro.operators.functionals._mse_data_discrepancy import mse_data_discrepancy


@pytest.mark.parametrize(
    'data, x, expected_mse',
    [
        ((0.0, 0.0), (0.0, 0.0), (0.0)),  # zero-tensors deliver 0-error
        ((0.0 + 1j * 0, 0.0), (0.0 + 1j * 0, 0.0), (0.0)),  # zero-tensors deliver 0-error; complex-valued
        ((1.0, 0.0), (1.0, 0.0), (0.0)),  # same tensors; both real-valued
        ((1.0, 0.0), (1.0 + 1j * 0, 0.0), (0.0)),  # same tensors; input complex-valued
        ((1.0, 0.0), (1.0 + 1j * 1, 0.0), (0.5)),  # different tensors; input complex-valued
        ((1.0 + 1j * 0, 0.0), (1.0, 0.0), (0.0)),  # same tensors; data complex-valued
        ((1.0 + 1j * 1, 0.0), (1.0, 0.0), (0.5)),  # different tensors; data complex-valued
        ((1.0 + 1j * 0, 0.0), (1.0 + 1j * 0, 0.0), (0.0)),  # same tensors; both complex-valued with imag part=0
        ((1.0 + 1j * 1, 0.0), (1.0 + 1j * 1, 0.0), (0.0)),  # same tensors; both complex-valued with imag part>0
        ((0.0 + 1j * 1, 0.0), (0.0 + 1j * 1, 0.0), (0.0)),  # same tensors; both complex-valued with real part=0
    ],
)
def test_mse_functional(data, x, expected_mse):
    """Test if mse_data_discrepancy matches expected values.

    Expected values are supposed to be
    1/N*|| . - data||_2^2
    """

    mse_dc = mse_data_discrepancy(torch.tensor(data))
    (mse,) = mse_dc(torch.tensor(x))
    torch.testing.assert_close(mse, torch.tensor(expected_mse))
