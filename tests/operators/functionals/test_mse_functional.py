"""Tests for MSE-functional."""

import pytest
import torch
from mrpro.operators.functionals.MSEDataDiscrepancy import MSEDataDiscrepancy


@pytest.mark.parametrize(
    ('data', 'x', 'expected_mse'),
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

    mse_op = MSEDataDiscrepancy(torch.tensor(data))
    (mse,) = mse_op(torch.tensor(x))
    torch.testing.assert_close(mse, torch.tensor(expected_mse))
